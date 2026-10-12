"""Contrats d'import exécutés en base, sans fournisseur externe."""
import json
import tempfile
import time as clock
from contextlib import contextmanager
from datetime import date, time
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase

from core.models import RecurringBlock, ScheduledBlock, Task, UploadedDocument
from services.agent_v2.importation import document_du_tour, message_import, resume_import
from services.document_processor import DocumentProcessor, GENERIC_PROCESSING_ERROR
from services.agent_v2.budget_tour import ouvrir_trace, fermer_trace
from services.agent_v2.trace_tour import trace_courante


class CorrectionImportsTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='import-correction')
        self.other = get_user_model().objects.create_user(username='import-autre')
        self.doc = UploadedDocument.objects.create(user=self.user, file_name='horaire.pdf')
        self.proc = DocumentProcessor()
        self.payload = {'courses': [
            {'name': 'Maths', 'day': 'lundi', 'start_time': '08:07', 'end_time': '10:52'},
            {'name': 'Chimie', 'day': 'mardi', 'start_time': '12:00', 'end_time': '14:00'},
        ]}

    @contextmanager
    def extraction(self, payload=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'horaire.pdf'
            path.write_bytes(b'%PDF-1.7\nlocal fixture')
            @contextmanager
            def fichier(document):
                yield str(path)
            with patch.object(self.proc, '_get_local_file', fichier), patch.object(
                self.proc, '_process_pdf_smart',
                return_value=(json.dumps(payload if payload is not None else self.payload), 'fixture'),
            ) as provider:
                yield provider

    def event(self, name='Match', **extra):
        return {'name': name, 'date': date.today().isoformat(),
                'start_time': '18:00', 'end_time': '20:00', **extra}

    def test_processed_reste_faux_pendant_les_ecritures(self):
        observed = []
        create = RecurringBlock.objects.create
        def observe(**kwargs):
            observed.append(UploadedDocument.objects.get(pk=self.doc.pk).processed)
            return create(**kwargs)
        with self.extraction(), patch.object(RecurringBlock.objects, 'create', side_effect=observe):
            self.proc.process_document(self.doc)
        self.assertEqual(observed, [False, False])
        self.doc.refresh_from_db()
        self.assertTrue(self.doc.processed)

    def test_erreur_deuxieme_ligne_annule_tout_et_reste_reessayable(self):
        create = RecurringBlock.objects.create
        def fail_second(**kwargs):
            if kwargs['title'] == 'Chimie':
                raise RuntimeError('secret-token-internal')
            return create(**kwargs)
        with self.extraction(), patch.object(RecurringBlock.objects, 'create', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, GENERIC_PROCESSING_ERROR):
                self.proc.process_document(self.doc)
        self.doc.refresh_from_db()
        self.assertFalse(self.doc.processed)
        self.assertEqual(self.doc.processing_error, GENERIC_PROCESSING_ERROR)
        self.assertFalse(RecurringBlock.all_objects.filter(user=self.user).exists())
        with self.extraction():
            self.proc.process_document(self.doc)
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 2)

    def test_echec_placement_annule_aussi_la_tache(self):
        with self.extraction({'events': [self.event()]}), patch.object(
            ScheduledBlock.objects, 'create', side_effect=RuntimeError('internal')
        ):
            with self.assertRaises(RuntimeError):
                self.proc.process_document(self.doc)
        self.assertFalse(Task.objects.filter(user=self.user).exists())
        self.assertFalse(ScheduledBlock.objects.filter(user=self.user).exists())

    def test_horaires_incomplets_restent_en_attente_sans_placement(self):
        for section in ('courses', 'shifts', 'events'):
            for times in ({'start_time': None}, {'end_time': None},
                          {'start_time': None, 'end_time': None}):
                with self.subTest(section=section, times=times):
                    doc = UploadedDocument.objects.create(user=self.user, file_name='incomplet.pdf')
                    data = {section: [self.event(**times)]}
                    self.proc._create_recurring_blocks(doc, data)
                    self.assertFalse(ScheduledBlock.objects.filter(user=self.user).exists())
                    self.assertFalse(Task.objects.filter(user=self.user).exists())
                    doc.refresh_from_db()
                    pending = doc.extracted_data.get('_pending_entries', [])
                    self.assertEqual(len(pending), 1)
                    self.assertTrue(pending[0]['missing_fields'])

    def test_recurrence_sans_fin_ne_recoit_aucune_heure_inventee(self):
        data = {'courses': [{'name': 'Maths', 'day': 'lundi', 'start_time': '08:00'}]}
        self.proc._create_recurring_blocks(self.doc, data)
        self.assertFalse(RecurringBlock.all_objects.filter(user=self.user).exists())
        self.doc.refresh_from_db()
        self.assertEqual(len(self.doc.extracted_data['_pending_entries']), 1)

    def test_date_invalide_ne_devient_pas_recurrence(self):
        self.proc._create_recurring_blocks(self.doc, {'events': [
            self.event(date='2026-02-30', day='lundi')
        ]})
        self.assertFalse(RecurringBlock.all_objects.filter(user=self.user).exists())
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.extracted_data['_pending_entries'][0]['missing_fields'], ['date'])

    def test_resume_exclut_les_placements_sans_provenance(self):
        task = Task.objects.create(user=self.user, title='Autre rendez-vous')
        ScheduledBlock.objects.create(user=self.user, task=task, date=date.today(),
                                      start_time='09:00', end_time='10:00', locked=True)
        self.assertEqual(resume_import(self.doc)['dates'], [])

    def test_piece_jointe_autre_compte_ou_instance_perimee_refusee(self):
        self.doc.processed = True
        self.doc.save(update_fields=['processed'])
        self.assertIsNone(document_du_tour(self.other, self.doc))
        UploadedDocument.objects.filter(pk=self.doc.pk).update(processed=False)
        self.assertIsNone(document_du_tour(self.user, self.doc))

    def test_cache_et_reimport_ne_reattribuent_pas_les_evenements(self):
        with self.extraction({'events': [self.event()]}) as provider:
            self.proc.process_document(self.doc)
            doc2 = UploadedDocument.objects.create(user=self.user, file_name='copie.pdf')
            self.proc.process_document(doc2)
            self.assertEqual(provider.call_count, 1)
            self.assertEqual(len(resume_import(self.doc)['dates']), 1)
            self.assertEqual(resume_import(doc2)['dates'], [])
            self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)
            self.proc.process_document(self.doc)
            self.assertEqual(provider.call_count, 1)
        self.assertEqual(ScheduledBlock.objects.get(user=self.user).source_document_id, self.doc.pk)

    def test_import_mixte_termine_et_explique_les_precisions(self):
        payload = {'courses': self.payload['courses'], 'events': [self.event(end_time=None)]}
        with self.extraction(payload):
            self.proc.process_document(self.doc)
        self.doc.refresh_from_db()
        self.assertTrue(self.doc.processed)
        summary = resume_import(self.doc)
        self.assertEqual(len(summary['blocs']), 2)
        self.assertEqual(summary['dates'], [])
        self.assertEqual(summary['en_attente'], 1)
        self.assertIn('fin', message_import(summary))
        self.assertNotIn('chevauche', message_import(summary))

    def test_recu_conserve_toutes_les_dates_sans_limite_arbitraire(self):
        with self.extraction({'events': [self.event(name=f'Match {i}') for i in range(41)]}):
            self.proc.process_document(self.doc)
        self.assertEqual(len(resume_import(self.doc)['dates']), 41)

    @contextmanager
    def budget(self):
        trace = ouvrir_trace(self.user, delai=60)
        token = trace_courante.set(trace)
        try:
            yield trace
        finally:
            fermer_trace(trace)
            trace_courante.reset(token)

    def test_fin_du_delai_annule_la_materialisation(self):
        create = RecurringBlock.objects.create
        with self.budget() as trace:
            def expire(**kwargs):
                block = create(**kwargs)
                trace.echeance = clock.monotonic() - 1
                return block
            with self.extraction(), patch.object(RecurringBlock.objects, 'create', side_effect=expire):
                with self.assertRaises(RuntimeError):
                    self.proc.process_document(self.doc)
        self.assertFalse(RecurringBlock.all_objects.filter(user=self.user).exists())
        self.doc.refresh_from_db()
        self.assertFalse(self.doc.processed)

    def test_trace_fermee_interdit_la_creation_meme_depuis_cache(self):
        with self.budget() as trace:
            fermer_trace(trace)
            with self.assertRaises(TimeoutError):
                self.proc._materialize_document(self.doc, self.payload)
        self.assertFalse(RecurringBlock.all_objects.filter(user=self.user).exists())

    def test_gemini_reserve_mesure_et_borne_la_requete(self):
        self.proc.gemini_model = 'fixture'
        self.proc.gemini_client = Mock()
        self.proc.gemini_client.models.generate_content.return_value = SimpleNamespace(
            text='{}', usage_metadata=SimpleNamespace(prompt_token_count=20,
                candidates_token_count=30, total_token_count=55))
        with self.budget() as trace:
            self.proc._call_gemini('texte')
        usage = trace.usages[-1]
        self.assertEqual((usage['phase'], usage['fournisseur'], usage['total']),
                         ('extraction', 'gemini', 55))
        config = self.proc.gemini_client.models.generate_content.call_args.kwargs['config']
        self.assertEqual(config.max_output_tokens, 4096)
        self.assertLessEqual(config.http_options.timeout, 60000)
        self.assertEqual(config.http_options.retry_options.attempts, 1)

    def test_gemini_usage_absent_ne_devient_pas_zero(self):
        self.proc.gemini_model = 'fixture'
        self.proc.gemini_client = Mock()
        self.proc.gemini_client.models.generate_content.return_value = SimpleNamespace(text='{}')
        with self.budget() as trace:
            self.proc._call_gemini('texte')
        self.assertFalse(trace.usages[-1]['usage_connu'])
        self.assertGreater(trace.total_jetons, 0)

    def test_deepseek_mesure_reponse_et_coupe_reessais_sdk(self):
        raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=13, total_tokens=25))
        provider = Mock()
        provider._request_kwargs.return_value = {'model': 'fixture', 'messages': []}
        provider._client.with_options.return_value.chat.completions.create.return_value = raw
        provider._parse_response.return_value = SimpleNamespace(text='{}', is_error=False)
        with self.budget() as trace, patch('services.llm.get_provider', return_value=provider):
            self.assertEqual(self.proc._structure_text_with_deepseek('texte'), '{}')
        self.assertEqual(trace.usages[-1]['total'], 25)
        self.assertEqual(provider.max_tokens, 4096)
        self.assertEqual(provider._client.with_options.call_args.kwargs['max_retries'], 0)

    def test_async_transmet_le_contexte_et_un_future(self):
        with self.budget() as trace, patch.object(self.proc, '_process_document_by_id',
                side_effect=lambda pk: {'trace': trace_courante.get(), 'pk': pk}), patch(
                'services.document_processor.run_in_background', side_effect=lambda f: f()):
            future = self.proc.process_document_async(self.doc.pk)
            self.assertIs(future.result()['trace'], trace)
            self.assertEqual(future.result()['pk'], self.doc.pk)
            self.assertFalse(trace.reservation.fermee)

    def test_rendu_precise_les_champs_sans_proposer_confirmation_impossible(self):
        from services.agent_v2.rendu import _recap_import
        self.proc._create_recurring_blocks(self.doc, {'events': [self.event(end_time=None)]})
        self.doc.refresh_from_db()
        rendered = _recap_import(resume_import(self.doc), date.today())
        self.assertIn("précise l'heure de fin", rendered)
        self.assertNotIn('confirme-le', rendered)

    def test_liens_incoherents_autre_utilisateur_exclus_du_recu(self):
        other_task = Task.objects.create(user=self.other, title='Privé')
        ScheduledBlock.objects.create(user=self.user, task=other_task, date=date.today(),
            start_time='09:00', end_time='10:00', source_document=self.doc)
        RecurringBlock.all_objects.create(user=self.other, title='Privé', day_of_week=0,
            start_time='09:00', end_time='10:00', source_document=self.doc)
        summary = resume_import(self.doc)
        self.assertEqual(summary['blocs'], [])
        self.assertEqual(summary['dates'], [])

    def test_retry_est_mesure_sans_secret_dans_les_logs(self):
        from requests import HTTPError, Response
        response = Response()
        response.status_code = 429
        failure = HTTPError('https://provider.invalid/?key=secret-fixture', response=response)
        self.proc.gemini_model = 'fixture'
        self.proc.gemini_client = Mock()
        self.proc.gemini_client.models.generate_content.side_effect = [failure, SimpleNamespace(
            text='{}', usage_metadata=SimpleNamespace(prompt_token_count=20,
                candidates_token_count=30, total_token_count=50))]
        with self.budget() as trace, patch('services.document_processor.time_module.sleep'), self.assertLogs(
                'services.document_processor', level='WARNING') as logs:
            self.proc._call_gemini('texte')
        self.assertEqual(len(trace.usages), 2)
        self.assertFalse(trace.usages[0]['usage_connu'])
        self.assertTrue(trace.usages[1]['usage_connu'])
        self.assertNotIn('secret-fixture', '\n'.join(logs.output))

    def test_retry_refuse_d_attendre_au_dela_du_delai(self):
        from requests import HTTPError, Response
        response = Response()
        response.status_code = 429
        self.proc.gemini_model = 'fixture'
        self.proc.gemini_client = Mock()
        self.proc.gemini_client.models.generate_content.side_effect = HTTPError('rate-limit', response=response)
        with self.budget() as trace, patch('services.document_processor.time_module.sleep') as sleep:
            trace.echeance = clock.monotonic() + 0.2
            with self.assertRaises(TimeoutError):
                self.proc._call_gemini('texte')
        sleep.assert_not_called()
        self.assertEqual(self.proc.gemini_client.models.generate_content.call_count, 1)

    def test_reprise_directe_ne_signale_pas_ses_propres_entrees_comme_ignorees(self):
        payload = {**self.payload, 'events': [self.event()]}
        self.proc._create_recurring_blocks(self.doc, payload)
        self.proc._create_recurring_blocks(self.doc, payload)
        self.doc.refresh_from_db()
        summary = resume_import(self.doc)
        self.assertEqual(summary['ignores'], [])
        self.assertEqual(len(summary['blocs']), 2)
        self.assertEqual(len(summary['dates']), 1)

    def test_cache_reste_non_traite_jusqua_fin_materialisation(self):
        with self.extraction():
            self.proc.process_document(self.doc)
            copy_doc = UploadedDocument.objects.create(user=self.user, file_name='copie.pdf')
            materialize = self.proc._create_recurring_blocks
            observed = []
            def inspect(document, data):
                observed.append(UploadedDocument.objects.get(pk=document.pk).processed)
                return materialize(document, data)
            with patch.object(self.proc, '_create_recurring_blocks', side_effect=inspect):
                self.proc.process_document(copy_doc)
        self.assertEqual(observed, [False])
        copy_doc.refresh_from_db()
        self.assertTrue(copy_doc.processed)

    def test_huggingface_mesure_la_tentative(self):
        from PIL import Image
        self.proc.hf_model = 'fixture'
        self.proc.hf_client = Mock()
        self.proc.hf_client.chat_completion.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        with self.budget() as trace:
            self.assertEqual(self.proc._call_hf_vision('texte', [Image.new('RGB', (8, 8))]), '{}')
        self.assertEqual(trace.usages[-1]['fournisseur'], 'huggingface')
        self.assertEqual(trace.usages[-1]['total'], 15)


class OutilDocumentEtatsTests(TestCase):
    def setUp(self):
        from services.agent.tools.documents import ProcessUploadedDocumentTool
        self.user = get_user_model().objects.create_user(username='doc-outil')
        self.other = get_user_model().objects.create_user(username='doc-outil-autre')
        self.tool = ProcessUploadedDocumentTool()

    def document(self, **kwargs):
        return UploadedDocument.objects.create(user=self.user, file_name='horaire.pdf', **kwargs)

    def test_absent_est_introuvable(self):
        result = self.tool.execute(self.user)
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertIn('Aucun document', result.message)

    def test_document_autre_compte_ne_revele_pas_son_etat(self):
        private = UploadedDocument.objects.create(user=self.other,
            file_name='prive.pdf', processed=True, extracted_data={'private': 'texte confidentiel'})
        result = self.tool.execute(self.user, document_id=private.pk)
        self.assertFalse(result.success)
        self.assertEqual(result.data, {})
        self.assertNotIn('confidentiel', result.message)
        self.assertNotIn('prive.pdf', result.message)

    def test_analyse_en_cours_sans_erreur(self):
        doc = self.document(processed=False)
        result = self.tool.execute(self.user, document_id=doc.pk)
        self.assertFalse(result.success)
        self.assertEqual(result.data['status'], 'processing')

    def test_erreur_prioritaire_sur_processing_et_message_sur(self):
        doc = self.document(processed=False, processing_error='https://provider.invalid/?key=secret-fixture')
        result = self.tool.execute(self.user, document_id=doc.pk)
        self.assertFalse(result.success)
        self.assertEqual(result.data['status'], 'failed')
        self.assertIn('échoué', result.message)
        self.assertNotIn('secret-fixture', str(result.data) + result.message)
        self.assertNotIn('encore en cours', result.message)

    def test_extraction_terminee_avec_heures_manquantes_demande_precision(self):
        pending = {'source': 'events', 'index': 0,
                   'entry': {'name': 'Match', 'date': date.today().isoformat(), 'start_time': '18:00'},
                   'missing_fields': ['end_time']}
        doc = self.document(processed=True, extracted_data={
            '_import_receipt_version': 1, '_pending_entries': [pending]})
        result = self.tool.execute(self.user, document_id=doc.pk)
        self.assertTrue(result.success)
        self.assertEqual(result.data['status'], 'needs_clarification')
        self.assertEqual(result.data['pending_count'], 1)
        self.assertEqual(result.data['pending_entries'], [pending])
        self.assertIn("précise l'heure de fin", result.message)
        self.assertNotIn('traité avec succès', result.message)
        self.assertNotIn('Horaire importé', result.message)

    def test_bloc_douteux_demande_confirmation_distincte(self):
        doc = self.document(processed=True)
        RecurringBlock.all_objects.create(user=self.user, source_document=doc,
            title='Cours', day_of_week=0, start_time='09:00', end_time='10:00',
            status=RecurringBlock.STATUS_PENDING)
        result = self.tool.execute(self.user, document_id=doc.pk)
        self.assertTrue(result.success)
        self.assertEqual(result.data['status'], 'needs_confirmation')
        self.assertEqual(result.data['pending_count'], 0)
        self.assertEqual(result.data['blocks_to_confirm'], 1)

    def test_termine_renvoie_les_creations_verifiees(self):
        doc = self.document(processed=True)
        RecurringBlock.objects.create(user=self.user, source_document=doc,
            title='Maths', day_of_week=0, start_time='09:00', end_time='10:00')
        result = self.tool.execute(self.user, document_id=doc.pk)
        self.assertTrue(result.success)
        self.assertEqual(result.data['status'], 'processed')
        self.assertEqual(result.data['pending_count'], 0)
        self.assertIn('Maths', result.message)
        self.assertEqual(len(result.data['import_summary']['blocs']), 1)


class ResolutionDocumentTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='doc-resolution')
        self.other = get_user_model().objects.create_user(username='doc-resolution-autre')
        self.doc = UploadedDocument.objects.create(user=self.user, file_name='matchs.pdf')
        self.proc = DocumentProcessor()
        self.data = {'events': [{'name': 'Match', 'date': date.today().isoformat()}]}
        self.proc._create_recurring_blocks(self.doc, self.data)
        self.doc.processed = True
        self.doc.save(update_fields=['processed'])

    def resolve(self, **fields):
        from services.agent.tools.documents import ResolveDocumentEntryTool
        return ResolveDocumentEntryTool().execute(self.user, document_id=self.doc.pk,
                                                  source='events', index=0, **fields)

    def test_contexte_recent_exclut_autre_evenement_et_presente_precision(self):
        from services.agent.agent import PlannerAgent
        task = Task.objects.create(user=self.user, title='Autre rendez-vous')
        ScheduledBlock.objects.create(user=self.user, task=task, date=date.today(),
                                      start_time='09:00', end_time='10:00', locked=True)
        text = PlannerAgent()._recent_import_context(self.user)
        self.assertNotIn('Autre rendez-vous', text)
        self.assertIn('Match', text)
        self.assertIn('start_time', text)
        self.assertIn('document_id', text)

    def test_resolution_partielle_conserve_la_precision_restante(self):
        result = self.resolve(start_time='18:00')
        self.assertTrue(result.success)
        self.assertEqual(result.data['resolution']['missing_fields'], ['end_time'])
        self.assertEqual(result.data['resolution']['entry']['start_time'], '18:00')
        self.assertFalse(ScheduledBlock.objects.filter(user=self.user).exists())
        self.assertNotIn('Planifié', result.message)

    def test_resolution_finale_lie_provenance_et_solde_la_precision(self):
        self.resolve(start_time='18:00')
        result = self.resolve(end_time='20:00')
        self.assertTrue(result.success)
        sb = ScheduledBlock.objects.get(source_document=self.doc)
        self.assertEqual(sb.task.title, 'Match')
        from django.utils import timezone
        self.assertTrue(timezone.is_aware(sb.task.deadline))
        self.assertEqual(timezone.localtime(sb.task.deadline).date(), sb.date)
        self.assertEqual(timezone.localtime(sb.task.deadline).time(), time.min)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.extracted_data['_pending_entries'], [])
        self.assertEqual(result.data['resolution']['targets'][0]['id'], sb.pk)

    def test_document_etranger_refuse(self):
        from services.agent.tools.documents import ResolveDocumentEntryTool
        result = ResolveDocumentEntryTool().execute(self.other, document_id=self.doc.pk,
            source='events', index=0, start_time='18:00', end_time='20:00')
        self.assertFalse(result.success)
        self.assertFalse(ScheduledBlock.objects.exists())
        self.assertNotIn('Match', result.message)

    def test_double_appel_ne_duplique_pas(self):
        first = self.resolve(start_time='18:00', end_time='20:00')
        again = self.resolve(start_time='18:00', end_time='20:00')
        self.assertTrue(first.success)
        self.assertTrue(again.success)
        self.assertTrue(again.data['already_resolved'])
        self.assertEqual(first.data['resolution'], again.data['resolution'])
        self.assertEqual(ScheduledBlock.objects.filter(source_document=self.doc).count(), 1)

    def test_erreur_placement_preserve_precision_et_aucune_tache_orpheline(self):
        with patch.object(ScheduledBlock.objects, 'create', side_effect=RuntimeError('secret-fixture')):
            result = self.resolve(start_time='18:00', end_time='20:00')
        self.assertFalse(result.success)
        self.assertNotIn('secret-fixture', result.message)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.extracted_data['_pending_entries'][0]['missing_fields'],
                         ['start_time', 'end_time'])
        self.assertFalse(Task.objects.filter(user=self.user).exists())

    def test_doublon_autre_document_ne_reattribue_pas_provenance(self):
        other_doc = UploadedDocument.objects.create(user=self.user, file_name='autre.pdf')
        task = Task.objects.create(user=self.user, title='Match')
        sb = ScheduledBlock.objects.create(user=self.user, task=task, date=date.today(),
            start_time='18:00', end_time='20:00', locked=True, source_document=other_doc)
        result = self.resolve(start_time='18:00', end_time='20:00')
        self.assertFalse(result.success)
        sb.refresh_from_db()
        self.assertEqual(sb.source_document_id, other_doc.pk)
        self.doc.refresh_from_db()
        self.assertEqual(len(self.doc.extracted_data['_pending_entries']), 1)

    def test_resolution_ne_materialise_pas_une_autre_entree_attendue(self):
        self.data['events'].append({'name': 'Gala', 'date': date.today().isoformat()})
        self.proc._create_recurring_blocks(self.doc, self.data)
        self.resolve(start_time='18:00', end_time='20:00')
        self.doc.refresh_from_db()
        self.assertEqual(len(self.doc.extracted_data['_pending_entries']), 1)
        self.assertEqual(self.doc.extracted_data['_pending_entries'][0]['index'], 1)
        self.assertEqual(list(ScheduledBlock.objects.values_list('task__title', flat=True)), ['Match'])

    def test_champ_connu_et_index_invalide_ne_sont_pas_modifies(self):
        from services.agent.tools.documents import ResolveDocumentEntryTool
        self.assertFalse(self.resolve(date='2099-01-01').success)
        result = ResolveDocumentEntryTool().execute(self.user, document_id=self.doc.pk,
            source='events', index=5, start_time='18:00')
        self.assertFalse(result.success)
        self.assertFalse(ScheduledBlock.objects.exists())

    def test_tool_reel_recu_relit_et_refuse_lecture_seule(self):
        from services.agent.tools import TOOL_MAP
        from services.agent_v2.outils import _Contexte, _executer_appel
        from services.agent_v2.registre import Registre
        from services.agent_v2.reconciliation import reconcilier
        from services.agent_v2.rendu import rendre_faits
        tool = TOOL_MAP['resolve_document_entry']
        params = {'document_id': self.doc.pk, 'source': 'events', 'index': 0,
                  'start_time': '18:00', 'end_time': '20:00'}
        readonly = Registre()
        readonly.lecture_seule = True
        _executer_appel(_Contexte(self.user, readonly, 'lecture', 'Mon document'), tool, params)
        self.assertFalse(ScheduledBlock.objects.exists())
        register = Registre()
        _executer_appel(_Contexte(self.user, register, 'précision', 'Le match est de 18h à 20h'), tool, params)
        reconcilier(self.user, register)
        action = register.actions[-1]
        self.assertTrue(action.est_mutation)
        self.assertTrue(action.etat_verifie)
        self.assertIn('Match', rendre_faits(register))
        ScheduledBlock.objects.filter(source_document=self.doc).update(end_time='21:00')
        reconcilier(self.user, register)
        self.assertFalse(register.actions[-1].etat_verifie)

    def test_date_precisee_conserve_son_annee_exacte(self):
        self.data['events'][0]['date'] = 'date illisible'
        self.proc._create_recurring_blocks(self.doc, self.data)
        result = self.resolve(date='2020-01-01', start_time='18:00', end_time='20:00')
        self.assertTrue(result.success)
        self.assertEqual(ScheduledBlock.objects.get(source_document=self.doc).date, date(2020, 1, 1))

    def test_jour_recurrent_manquant_est_resolu_sans_evenement_date(self):
        self.data = {'courses': [{'name': 'Maths', 'start_time': '08:00', 'end_time': '09:00'}]}
        self.proc._create_recurring_blocks(self.doc, self.data)
        from services.agent.tools.documents import ResolveDocumentEntryTool
        result = ResolveDocumentEntryTool().execute(self.user, document_id=self.doc.pk,
            source='courses', index=0, day='lundi')
        self.assertTrue(result.success)
        self.assertFalse(ScheduledBlock.objects.exists())
        self.assertEqual(RecurringBlock.all_objects.get(source_document=self.doc).day_of_week, 0)

    def test_date_peut_preciser_un_jour_absent_sans_creer_de_recurrence(self):
        self.data['events'][0].pop('date')
        self.proc._create_recurring_blocks(self.doc, self.data)
        result = self.resolve(date='2030-12-01', start_time='18:00', end_time='20:00')
        self.assertTrue(result.success)
        self.assertFalse(RecurringBlock.all_objects.exists())
        self.assertEqual(ScheduledBlock.objects.get(source_document=self.doc).date, date(2030, 12, 1))

    def test_tour_ferme_refuse_la_materialisation(self):
        trace = ouvrir_trace(self.user)
        token = trace_courante.set(trace)
        try:
            fermer_trace(trace)
            result = self.resolve(start_time='18:00', end_time='20:00')
        finally:
            trace_courante.reset(token)
        self.assertFalse(result.success)
        self.assertFalse(ScheduledBlock.objects.exists())
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.extracted_data['_pending_entries'][0]['missing_fields'],
                         ['start_time', 'end_time'])

    def test_sixieme_resolution_creatrice_attend_confirmation(self):
        from services.agent.tools import TOOL_MAP
        from services.agent.tools.base import ToolResult
        from services.agent_v2.outils import _Contexte, _executer_appel
        from services.agent_v2.registre import Registre
        register = Registre()
        for index in range(5):
            register.ajouter('create_task', {}, ToolResult(success=True,
                data={'task': {'id': 100 + index, 'title': f'Tâche {index}'}}))
        _executer_appel(_Contexte(self.user, register, 'limite', 'Le match est de 18h à 20h'),
            TOOL_MAP['resolve_document_entry'], {'document_id': self.doc.pk, 'source': 'events',
                'index': 0, 'start_time': '18:00', 'end_time': '20:00'})
        self.assertFalse(ScheduledBlock.objects.exists())
        self.assertEqual(register.actions[-1].donnees['demande']['motif'], 'creation_en_masse')


class ResolutionDocumentChargeurTests(TransactionTestCase):
    def test_decouverte_lecture_resolution_indirecte_et_recu(self):
        import asyncio
        from services.agent_v2.outils import outils_pour_le_modele, NOM_APPEL, NOM_CHERCHEUR
        from services.agent_v2.registre import Registre
        from services.agent_v2.reconciliation import reconcilier
        from services.agent_v2.rendu import rendre_faits
        user = get_user_model().objects.create_user(username='doc-indirect')
        doc = UploadedDocument.objects.create(user=user, file_name='match.pdf', processed=True)
        DocumentProcessor()._create_recurring_blocks(doc, {'events': [
            {'name': 'Match', 'date': date.today().isoformat()}]})
        register = Registre()
        tools = {t.name: t for t in outils_pour_le_modele(user, register,
            message_du_tour='Le match est de 18h à 20h', tache='doc:indirect')}
        found = asyncio.run(tools[NOM_CHERCHEUR].function_schema.function(
            besoin='resolve_document_entry'))
        self.assertIn('resolve_document_entry', found)
        read = asyncio.run(tools[NOM_APPEL].function_schema.function(
            nom='process_uploaded_document', parametres={'document_id': doc.pk}))
        self.assertIn('start_time', read)
        asyncio.run(tools[NOM_APPEL].function_schema.function(nom='resolve_document_entry',
            parametres={'document_id': doc.pk, 'source': 'events', 'index': 0,
                       'start_time': '18:00', 'end_time': '20:00'}))
        reconcilier(user, register)
        self.assertEqual(ScheduledBlock.objects.filter(source_document=doc).count(), 1)
        self.assertTrue(register.actions[-1].etat_verifie)
        self.assertIn('Match', rendre_faits(register))
