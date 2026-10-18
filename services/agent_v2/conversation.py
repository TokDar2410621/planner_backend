"""Entree publique V2 : style OpenCode, pas de decideur pre-boucle.

Le message et l'historique vont directement a la boucle. Le tap structure
passe par le code ; le budget journalier epuise rend sans appeler le modele.
"""
from __future__ import annotations

import logging
import time

from django.utils import timezone

from core.models import ConversationMessage
from services.agent_v2.agent import (
    PlannerAgentV2, _budget_jour_epuise,
    prose_budget_jour,
)
from services.agent_v2.redaction import sans_tiret_long

logger = logging.getLogger(__name__)


class AgentConversation(PlannerAgentV2):
    """Entree publique V2 : style OpenCode, pas de decideur pre-boucle."""

    def process_message_stream(self, user, message, attachment=None, *,
                               use_streaming=True, generate_quick_replies=False, tap=None,
                               conversation_id=None):
        from services.agent_v2.trace_tour import TraceTour, trace_courante
        from services.agent_v2.budget_tour import ouvrir_trace, fermer_trace, BudgetIndisponible
        self._trace = trace_courante.get()
        proprietaire = self._trace is None
        self._admission_refusee = False
        if proprietaire:
            try:
                self._trace = ouvrir_trace(user)
            except BudgetIndisponible:
                self._admission_refusee = True
                self._trace = TraceTour(time.monotonic())
        elif self._trace.reservation is None or self._trace.reservation.user_id != user.pk:
            self._admission_refusee = True
        jeton = trace_courante.set(self._trace)
        try:
            yield from self._traiter_tour(user, message, attachment,
                use_streaming=use_streaming, generate_quick_replies=generate_quick_replies, tap=tap,
                conversation_id=conversation_id)
        finally:
            try:
                if proprietaire:
                    fermer_trace(self._trace)
            finally:
                trace_courante.reset(jeton)

    def _traiter_tour(self, user, message, attachment=None, *,
                               use_streaming=True, generate_quick_replies=False, tap=None,
                               conversation_id=None):
        self.user = user
        self._depart_admission = time.perf_counter()
        self._decision_tour = None
        self._usage_decision = {}
        self._conversation = self._resoudre_conversation(user, conversation_id)
        # Le raccourci exige une option d'une demande recente du meme utilisateur.
        if not self._admission_refusee and self._tap_valide(user, tap):
            yield from super().process_message_stream(
                user, message, attachment, use_streaming=use_streaming,
                generate_quick_replies=False, tap=tap,
                conversation_id=conversation_id)
            return

        if self._admission_refusee or _budget_jour_epuise(user):
            # Garde-fou financier : sans budget, pas d'appel LLM. Rendu
            # direct, sans boucle.
            courant = ConversationMessage.objects.create(
                user=user, role='user', content=message, conversation=self._conversation)
            courant.attachments.set(getattr(self, '_chat_attachments', []) or ([attachment] if attachment else []))
            texte = sans_tiret_long(prose_budget_jour(user))
            metadata = {'agent': 'v2', 'contrat_tour': 'budget',
                        'en_reponse_a': courant.pk, 'actions': [], 'quick_replies': [],
                        'question_posee': False, 'usage': []}
            assistant = ConversationMessage.objects.create(
                user=user, role='assistant', content=texte, metadata=metadata,
                conversation=self._conversation)
            yield {'type': 'delta', 'text': texte}
            yield {'type': 'done', 'response': texte, 'quick_replies': [],
                   'assistant_message_id': assistant.pk,
                   'conversation_id': self._conversation.pk if self._conversation else None,
                   'blocks_created': [], 'tasks_created': [], 'raisonnement': '',
                   'question_posee': False, 'question': '', 'question_motif': ''}
            return

        # Style OpenCode : pas de decideur pre-boucle. Le message et
        # l'historique vont directement au modele, qui tranche lui-meme
        # (agir, demander via l'outil, repondre). Aucune classification
        # prealable, aucun routage hors boucle.
        yield from super().process_message_stream(
            user, message, attachment, use_streaming=use_streaming,
            generate_quick_replies=generate_quick_replies, tap=tap,
            conversation_id=conversation_id)

    @staticmethod
    def _tap_valide(user, tap):
        from services.agent_v2.demandes import _moment, _ids_options, FENETRE_ATTENTE
        from datetime import timedelta
        if not isinstance(tap, dict) or not tap.get('demande') or not tap.get('option'):
            return False
        lignes = list(ConversationMessage.objects.filter(user=user).order_by('-pk')[:2])
        if len(lignes) != 2 or lignes[0].role != 'assistant' or lignes[1].role != 'user':
            return False
        meta = lignes[0].metadata or {}
        if meta.get('en_reponse_a') != lignes[1].pk:
            return False
        maintenant = timezone.now()
        for demande in meta.get('demandes') or []:
            if not isinstance(demande, dict):
                continue
            emise = _moment(demande.get('emise_le'))
            if (emise and maintenant - FENETRE_ATTENTE < emise <= maintenant + timedelta(minutes=1)
                    and demande.get('cle') == tap['demande'] and tap['option'] in _ids_options(demande)):
                return True
        return False

    def quick_replies_for(self, user, user_message, assistant_response):
        """Les choix utiles sont deja produits dans le tour, sans second LLM."""
        chip = memoire.chip_inference(user, user_message or '')
        return [chip] if chip else []
