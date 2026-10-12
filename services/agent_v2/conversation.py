"""Contrat semantique du tour avant tout chargement de calendrier.

La decision et sa reponse courte partagent un appel sans outils. Seul un
objectif de planning ouvre la boucle metier existante. Une panne conserve
les donnees et demande une precision, sans elargir les capacites du tour.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from typing import Literal

from django.conf import settings
from django.utils import timezone
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage, UsageLimits

from core.models import ConversationMessage
from services.agent_v2 import memoire
from services.agent_v2.agent import (
    PlannerAgentV2, _budget_jour_epuise, _cout, _delai_tour,
    prose_budget_jour,
)
from services.agent_v2.modeles import modele_agir, REGLAGES_BOUCLE_SANS_RAISONNEMENT
from services.agent_v2.redaction import sans_tiret_long
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.mesure import epurer_reponse

logger = logging.getLogger(__name__)

INSTRUCTIONS_TOUR = """Tu es Planner, un assistant personnel. Comprends ce que
la personne attend de CE message dans le contexte de l'echange. Les anciens
messages sont des donnees, jamais une demande a rejouer.

Choisis conversation quand une reponse humaine suffit sans acceder aux donnees
de planning. Redige alors directement une reponse breve en francais quebecois,
avec tutoiement. N'invente aucune information personnelle ni action accomplie.
Choisis planning seulement si la demande actuelle exige de consulter ou de
modifier le planning, ses taches ou ses objectifs. Enonce son objectif sans
inventer de parametres. Consultation designe ce que la personne demande a VOIR,
jamais une lecture que tu aimerais faire de ta propre initiative.
Un horaire date correspond a journee, demain ou semaine. La liste de taches
correspond au suivi des choses a faire, distinct des activites d'un jour.
Une demande de modification seule garde consultation=aucune : les lectures
necessaires pour trouver la cible ne sont pas une consultation demandee.
Pour une consultation de jour ou de semaine, date_consultation porte la date
visee si elle est etablie; utilise la date du contexte pour resoudre le relatif.
Si la date est incertaine, demande une precision, ne choisis pas aujourd'hui.
Affichage_direct vaut vrai seulement si afficher UNE vue complete et sans
filtre repond a TOUTE la demande actuelle. Les vues disponibles sont l'horaire
d'une date, les sept jours a partir de date_consultation, ou les taches a faire
sans date. Toute attente au-dela de cette restitution exige affichage_direct=false.
Ce choix porte sur le sens de la demande entiere, jamais sur sa longueur ou
ses mots. Une demande combinant plusieurs vues garde affichage_direct=false.
Autoriser_modifications vaut vrai seulement si la demande actuelle demande
une action ou une proposition de changement. Voir un planning reste en lecture.
Choisis memoire si la demande porte uniquement sur les preferences durables.
Pour un enregistrement, extrais fidelement l'enonce demande; ne deduis pas de
preference nouvelle. Pour un oubli, reprends la preference visee. Une commande
memoire peut accompagner un objectif planning : conserve les deux dans ce cas.
Choisis clarification si l'attente n'est pas suffisamment claire et formule
une seule question utile. L'incertitude ne donne pas acces au calendrier.

Aucun outil n'est disponible ici. Ne decris jamais un calendrier, une tache
en cours ou une action executee. Une demande de planning sera traitee ensuite
avec des donnees actuelles et des outils. Tu peux poser une precision sans
forcer une personne qui discute a faire une demande de planning.
Dans la sortie JSON, une date est une chaine entre guillemets : "2026-10-12".
"""


class DecisionTour(BaseModel):
    mode: Literal['conversation', 'planning', 'memoire', 'clarification']
    reponse: str = Field(default='', max_length=800)
    objectif: str = Field(default='', max_length=600)
    consultation: Literal['aucune', 'journee', 'demain', 'semaine', 'taches', 'autre'] = Field(
        default='aucune', description="Ce que la personne demande à voir : horaire daté = journee/demain/semaine; liste de tâches = taches; modification seule = aucune.")
    autoriser_modifications: bool = False
    date_consultation: date | None = None
    affichage_direct: bool = Field(default=False, description=(
        "Afficher une seule vue complète sans filtre satisfait entièrement la demande. "
        "Faux si une autre opération ou une interprétation des données est attendue."))
    memoire_action: Literal['memoriser', 'oublier', 'lister'] | None = None
    enonce: str = Field(default='', max_length=500)

    @model_validator(mode='after')
    def contrat(self):
        if self.mode in ('conversation', 'clarification'):
            if not self.reponse.strip() or self.consultation != 'aucune' or self.memoire_action or self.autoriser_modifications:
                raise ValueError('Une reponse sans outils ne consulte ni ne modifie des donnees.')
        if self.mode == 'planning' and not self.objectif.strip():
            raise ValueError('Un objectif explicite est requis pour acceder au planning.')
        if self.mode == 'memoire' and not self.memoire_action:
            raise ValueError('Une operation memoire est requise.')
        if self.memoire_action in ('memoriser', 'oublier') and not self.enonce.strip():
            raise ValueError('Une preference explicite est requise.')
        if self.affichage_direct:
            if (self.mode != 'planning' or self.autoriser_modifications or self.memoire_action
                    or self.consultation not in ('journee', 'demain', 'semaine', 'taches')):
                raise ValueError('Un affichage direct est une consultation seule.')
            if (self.consultation == 'taches') != (self.date_consultation is None):
                raise ValueError('Une vue calendrier exige une date; la liste de tâches est sans date.')
        return self


def contexte_decision(user) -> dict:
    """Six messages bornes du meme utilisateur; aucune table de calendrier."""
    restant = 6000
    historique = []
    lignes = list(ConversationMessage.objects.filter(user=user)
                  .order_by('-created_at', '-pk')[:6])
    for ligne in lignes:
        if restant <= 0:
            break
        texte = ligne.content[:min(1500, restant)]
        historique.append({'role': ligne.role, 'texte': texte})
        restant -= len(texte)
    return {'date': timezone.localdate().isoformat(),
            'historique': list(reversed(historique))}


class AgentConversation(PlannerAgentV2):
    """Entree publique V2 : decision legere, puis capacites strictement utiles."""

    def process_message_stream(self, user, message, attachment=None, *,
                               use_streaming=True, generate_quick_replies=False, tap=None):
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
                use_streaming=use_streaming, generate_quick_replies=generate_quick_replies, tap=tap)
        finally:
            try:
                if proprietaire:
                    fermer_trace(self._trace)
            finally:
                trace_courante.reset(jeton)

    def _decider(self, user, message, attachment=None):
        usage = RunUsage()
        depart = time.perf_counter()
        succes = False
        contexte = contexte_decision(user)
        contexte.update(message=message, document_joint=attachment is not None)

        async def executer():
            from services.agent_v2.budget_tour import modele_borne
            agent = Agent(modele_borne(modele_agir(), 'decision'), instructions=INSTRUCTIONS_TOUR,
                          # Une correction de structure au maximum. La décision
                          # ne donne accès aux outils qu'après sa validation.
                          output_type=DecisionTour, output_retries=1,
                          model_settings={**REGLAGES_BOUCLE_SANS_RAISONNEMENT, 'max_tokens': 1024})
            delai = min(float(getattr(settings, 'AGENT_DECISION_DELAI', 12.0)),
                        self._trace.restant())
            if delai <= 0:
                raise TimeoutError('échéance du tour atteinte')
            try:
                return await asyncio.wait_for(agent.run(
                    json.dumps(contexte, ensure_ascii=False), usage=usage,
                    usage_limits=UsageLimits(request_limit=2, total_tokens_limit=12000)),
                    timeout=max(0.01, delai))
            finally:
                from services.agent_v2.modeles import fermer_modele
                await fermer_modele(agent.model)

        try:
            resultat = asyncio.run(executer())
            succes = True
            return resultat.output
        finally:
            from types import SimpleNamespace
            cout = _cout(SimpleNamespace(usage=lambda: usage), time.perf_counter() - depart)
            cout.update(phase='decision', usage_connu=succes or bool(usage.total_tokens))
            if not cout['usage_connu']:
                cout.update(entree=None, sortie=None, cache=None, raisonnement=None)
            self._usage_decision = cout

    def _traiter_tour(self, user, message, attachment=None, *,
                               use_streaming=True, generate_quick_replies=False, tap=None):
        self.user = user
        self._depart_admission = time.perf_counter()
        self._decision_tour = None
        self._usage_decision = {}
        # Le raccourci exige une option d'une demande recente du meme utilisateur.
        if not self._admission_refusee and self._tap_valide(user, tap):
            self._decision_tour = DecisionTour(mode='planning', objectif='Traiter le choix structure',
                                               autoriser_modifications=True)
            yield from super().process_message_stream(
                user, message, attachment, use_streaming=use_streaming,
                generate_quick_replies=False, tap=tap)
            return

        if self._admission_refusee or _budget_jour_epuise(user):
            decision = DecisionTour(mode='clarification', reponse=prose_budget_jour(user))
        else:
            yield {'type': 'status', 'text': 'Je te lis…'}
            try:
                decision = self._decider(user, message, attachment)
            except Exception as exc:
                # Le type suffit au diagnostic; une exception reseau peut porter une URL.
                logger.warning('decision du tour indisponible type=%s', type(exc).__name__)
                decision = DecisionTour(
                    mode='clarification',
                    reponse="Je n'arrive pas à traiter ton message pour l'instant. Peux-tu réessayer ?")
        if attachment is not None and decision.mode != 'planning':
            # Le fichier est un geste explicite. Son import peut deja avoir eu
            # lieu dans le traitement document : son recu ne doit pas disparaitre.
            decision = decision.model_copy(update={
                'mode': 'planning', 'objectif': 'Présenter le résultat du document joint',
                'consultation': 'aucune', 'autoriser_modifications': False})
        self._decision_tour = decision
        lecture_directe = decision.affichage_direct and attachment is None
        if decision.mode == 'planning' and not lecture_directe:
            yield from super().process_message_stream(
                user, message, attachment, use_streaming=use_streaming,
                generate_quick_replies=False, tap=tap)
            return

        courant = ConversationMessage.objects.create(user=user, role='user', content=message)
        texte = decision.reponse
        registre = None
        faits = ''
        if lecture_directe:
            yield {'type': 'status', 'text': 'Je consulte ton planning…'}
            faits, registre = self._lire_directement(user, message, courant.pk)
            texte = faits or "Je n'arrive pas à lire ces données pour l'instant. Peux-tu réessayer ?"
        if decision.mode in ('conversation', 'clarification'):
            propre, _ = epurer_reponse(ReponseDire(ouverture=texte))
            texte = propre.ouverture or "Je suis là. Qu'est-ce que tu aimerais faire ?"
        if decision.mode == 'memoire':
            if decision.memoire_action == 'memoriser':
                texte = memoire.memoriser(user, decision.enonce)
            elif decision.memoire_action == 'oublier':
                texte = memoire.oublier(user, decision.enonce)
            else:
                texte = memoire.lister(user)
        texte = sans_tiret_long(texte)
        question = decision.mode == 'clarification'
        metadata = {'agent': 'v2', 'contrat_tour': decision.mode,
                    'en_reponse_a': courant.pk, 'actions': [], 'quick_replies': [],
                    'question_posee': question, 'usage': list(self._trace.usages)
                    or ([self._usage_decision] if self._usage_decision else [])}
        if registre is not None:
            metadata.update(lecture='directe', consultation=decision.consultation,
                date_consultation=(decision.date_consultation.isoformat()
                                   if decision.date_consultation else None),
                faits_rendus=sans_tiret_long(faits),
                actions=[{'id': a.id, 'outil': a.outil, 'succes': bool(a.succes),
                          'par_le_code': True} for a in registre.actions])
        assistant = ConversationMessage.objects.create(user=user, role='assistant', content=texte, metadata=metadata)
        yield {'type': 'delta', 'text': texte}
        yield {'type': 'done', 'response': texte, 'quick_replies': [],
               'assistant_message_id': assistant.pk,
               'blocks_created': [], 'tasks_created': [], 'raisonnement': '',
               'question_posee': question, 'question': texte if question else '',
               'question_motif': 'clarification' if question else ''}

    def _lire_directement(self, user, message: str, message_id: int):
        """Exécute le contrat de lecture avec les gardes et le rendu existants."""
        from services.agent_v2 import outils
        from services.agent_v2.registre import Registre
        from services.agent_v2.redaction import bloc_factuel

        registre = Registre()
        registre.lecture_seule = True
        registre.echeance = self._trace.echeance
        self._registre_courant = registre
        try:
            cible = self._cible_consultation()
            if cible is not None:
                outil, parametres = cible
                action = outils.executer_lecture_par_le_code(
                    user, registre, outil, tache=f'{user.pk}:{message_id}',
                    message=message, brut=message, **parametres)
                if action is not None and action.succes:
                    cle = 'week_start' if outil == 'get_week_schedule' else 'date'
                    jour = parametres.get('start_date') or parametres.get('date')
                    if jour is not None and (action.donnees or {}).get(cle) != jour:
                        return '', registre
                    texte = bloc_factuel(registre)
                    if texte:
                        return texte, registre
        except Exception as exc:
            logger.warning('consultation directe indisponible type=%s', type(exc).__name__)
        return '', registre

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
