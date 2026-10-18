"""Réserve avant toute requête payante et rapproche une seule fois son usage.

Une réservation abandonnée par un processus reste bloquée pour cette journée.
Elle ne peut pas être remboursée sans savoir si le fournisseur a facturé.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, time as heure, timedelta
from contextlib import contextmanager, asynccontextmanager
import json
import logging
import threading
import time

import anyio

from django.conf import settings
from django.db import transaction, connections
from django.db.models import F
from django.utils import timezone

from services.agent_v2.trace_tour import TraceTour, trace_courante

logger = logging.getLogger(__name__)


class BudgetIndisponible(RuntimeError):
    pass


class BudgetEpuise(BudgetIndisponible):
    def __init__(self, message, motif='garde_cout'):
        super().__init__(message)
        self.motif = motif


def _entier(nom, defaut):
    try:
        return max(1, int(getattr(settings, nom, defaut)))
    except (ValueError, TypeError):
        return defaut


def _plafond_sortie(phase):
    """Plafond de jetons de sortie selon la phase.

    L'extraction de documents (phase='extraction') produit un JSON long et
    ne doit pas être étranglée par le plafond du chat (AGENT_MAX_JETONS_SORTIE,
    réglé bas en production). Un plafond dédié EXTRACTION_MAX_JETONS_SORTIE
    permet de l'ajuster sans toucher au chat.
    """
    if phase == 'extraction':
        return _entier('EXTRACTION_MAX_JETONS_SORTIE', 4096)
    return _entier('AGENT_MAX_JETONS_SORTIE', 4096)


def borne_texte(*parties):
    """Estimation prudente : environ trois octets UTF-8 par jeton.

    Aucun contenu n'est conservé. Images et fichiers exigent une borne dédiée.
    """
    octets = sum(len((p if isinstance(p, str) else json.dumps(
        p, ensure_ascii=False, default=str)).encode('utf-8')) for p in parties)
    return 1024 + (octets + 2) // 3


def _models():
    from core.models import BudgetJetonsJournalier, ReservationBudgetJetons
    return BudgetJetonsJournalier, ReservationBudgetJetons


def plafond_garde_cout(user, plafond_mesure):
    """Plafond interne distinct du quota affiché, configurable par catégorie."""
    from core.anonyme import est_anonyme
    nom = ('AGENT_V2_GARDE_COUT_JOUR_ANON' if est_anonyme(user)
           else 'AGENT_V2_GARDE_COUT_JOUR')
    return _entier(nom, max(plafond_mesure * 2, plafond_mesure + 120000))


def statut_usage(user):
    """Le contrat public du quota : aucune provision ne gonfle la jauge."""
    from services.agent_v2.agent import _budget_jetons_jour
    Budget, _ = _models()
    jour = timezone.localdate()
    ligne = Budget.objects.filter(user=user, jour=jour).first()
    limite = _budget_jetons_jour(user)
    garde = plafond_garde_cout(user, limite)
    utilises = ligne.jetons_mesures if ligne else 0
    provisionnes = (ligne.jetons + ligne.jetons_reserves) if ligne else 0
    motif = ('quota_mesure' if utilises >= limite else
             'garde_cout' if provisionnes >= garde else None)
    retour = timezone.make_aware(
        datetime.combine(jour + timedelta(days=1), heure.min),
        timezone.get_current_timezone(),
    )
    return {
        'jour': jour.isoformat(),
        'jetons_mesures': utilises,
        'limite_jetons': limite,
        'appels_non_mesures': ligne.appels_non_mesures if ligne else 0,
        'mesure_partielle': ligne.mesure_partielle if ligne else False,
        'bloque_par': motif,
        'reprise_a': retour.isoformat(),
    }


class ReservationTour:
    def __init__(self, ligne, plafond, garde_plafond):
        self.pk = ligne.pk
        self.budget_id = ligne.budget_id
        self.user_id = ligne.budget.user_id
        self.plafond = plafond
        self.garde_plafond = garde_plafond
        self.disponible = ligne.jetons_reserves
        self.fermee = False
        self.verrou = threading.RLock()

    def debiter(self, montant):
        """L'UPDATE conditionnel sérialise l'admission sur SQLite et PostgreSQL."""
        Budget, Reservation = _models()
        with self.verrou:
            if self.fermee:
                raise BudgetIndisponible('tour fermé')
            supplement = max(0, montant - self.disponible)
            try:
                with transaction.atomic():
                    if not Budget.objects.filter(
                            pk=self.budget_id,
                            jetons_mesures__lt=self.plafond,
                            jetons__lte=self.garde_plafond - F('jetons_reserves') - supplement,
                    ).update(jetons_reserves=F('jetons_reserves') + supplement):
                        budget = Budget.objects.get(pk=self.budget_id)
                        motif = ('quota_mesure' if budget.jetons_mesures >= self.plafond
                                 else 'garde_cout')
                        raise BudgetEpuise('budget journalier insuffisant', motif)
                    if not Reservation.objects.filter(pk=self.pk, cloturee=False).update(
                            jetons_reserves=F('jetons_reserves') + supplement - montant,
                            jetons_factures=F('jetons_factures') + montant):
                        raise BudgetIndisponible('réservation fermée')
                    Budget.objects.filter(pk=self.budget_id).update(
                        jetons_reserves=F('jetons_reserves') - montant,
                        jetons=F('jetons') + montant)
            except BudgetIndisponible:
                raise
            except Exception as exc:
                raise BudgetIndisponible('réservation indisponible') from exc
            self.disponible += supplement - montant

    def rapprocher(self, montant, reel):
        """Un retour connu rembourse sa provision, même après la fin du flux."""
        Budget, Reservation = _models()
        ecart = reel - montant
        with self.verrou, transaction.atomic():
            Budget.objects.filter(pk=self.budget_id).update(
                jetons=F('jetons') + ecart,
                jetons_mesures=F('jetons_mesures') + reel,
            )
            Reservation.objects.filter(pk=self.pk).update(
                jetons_factures=F('jetons_factures') + ecart)

    def noter_usage_inconnu(self):
        Budget, _ = _models()
        with self.verrou:
            Budget.objects.filter(pk=self.budget_id).update(
                appels_non_mesures=F('appels_non_mesures') + 1)

    def fermer(self):
        Budget, Reservation = _models()
        with self.verrou:
            if self.fermee:
                return
            with transaction.atomic():
                Budget.objects.select_for_update().get(pk=self.budget_id)
                # CAS : deux fermetures, même avec deux objets, ne remboursent pas deux fois.
                montant = Reservation.objects.get(pk=self.pk).jetons_reserves
                if Reservation.objects.filter(pk=self.pk, cloturee=False).update(
                        cloturee=True, jetons_reserves=0):
                    Budget.objects.filter(pk=self.budget_id).update(
                        jetons_reserves=F('jetons_reserves') - montant)
            self.disponible = 0
            self.fermee = True


def ouvrir_trace(user, delai=None):
    from services.agent_v2.agent import _budget_jetons_jour, _delai_tour
    Budget, Reservation = _models()
    plafond = _budget_jetons_jour(user)
    if plafond <= 0:
        raise BudgetEpuise('budget désactivé', 'quota_mesure')
    garde_plafond = plafond_garde_cout(user, plafond)
    montant = min(garde_plafond, _entier('AGENT_RESERVATION_INITIALE', 12000))
    trace = TraceTour(time.monotonic() + (_delai_tour() if delai is None else delai))
    try:
        budget, _ = Budget.objects.get_or_create(user=user, jour=timezone.localdate())
        if budget.jetons_mesures >= plafond:
            raise BudgetEpuise('quota journalier mesuré atteint', 'quota_mesure')
        montant = min(montant, garde_plafond - budget.jetons - budget.jetons_reserves)
        if montant <= 0:
            raise BudgetEpuise('garde de coût journalier atteint')
        with transaction.atomic():
            if not Budget.objects.filter(pk=budget.pk,
                    jetons_mesures__lt=plafond,
                    jetons__lte=garde_plafond - F('jetons_reserves') - montant).update(
                        jetons_reserves=F('jetons_reserves') + montant):
                budget.refresh_from_db()
                motif = ('quota_mesure' if budget.jetons_mesures >= plafond
                         else 'garde_cout')
                raise BudgetEpuise('budget journalier insuffisant', motif)
            ligne = Reservation.objects.create(budget=budget, jetons_reserves=montant)
    except BudgetIndisponible:
        raise
    except Exception as exc:
        raise BudgetIndisponible('admission indisponible') from exc
    trace.reservation = ReservationTour(ligne, plafond, garde_plafond)
    return trace


def fermer_trace(trace):
    reservation = getattr(trace, 'reservation', None)
    if reservation is not None:
        try:
            reservation.fermer()
        except Exception as exc:
            # Aucun remboursement optimiste en cas d'échec de la base.
            logger.warning('fermeture budget indisponible type=%s', type(exc).__name__)


class AppelFournisseur:
    def __init__(self, phase, fournisseur, jetons_entree, max_sortie):
        self.trace = trace_courante.get()
        self.reservation = getattr(self.trace, 'reservation', None)
        if self.reservation is None:
            raise BudgetIndisponible('aucune réservation pour cet appel')
        self.delai = self.trace.restant()
        if self.delai <= 0:
            raise TimeoutError('échéance du tour atteinte')
        self.max_sortie = min(max(1, int(max_sortie)), _plafond_sortie(phase))
        self.provision = max(1, int(jetons_entree)) + self.max_sortie
        if self.provision > _entier('AGENT_MAX_JETONS_APPEL', 120000):
            raise BudgetEpuise('contexte fournisseur trop volumineux', 'appel_trop_volumineux')
        self.phase, self.fournisseur = phase, fournisseur
        self.depart = time.monotonic()
        self.termine = False
        self.verrou = threading.Lock()
        self.reservation.debiter(self.provision)
        self.delai = self.trace.restant()
        if self.delai <= 0:
            # L'admission en base a consommé le délai, aucun envoi n'a commencé.
            self.reservation.rapprocher(self.provision, 0)
            raise TimeoutError('échéance du tour atteinte')
        self.index = self.trace.ajouter({'phase': self.phase, 'fournisseur': self.fournisseur,
            'usage_connu': False, 'entree': None, 'sortie': None, 'total': None,
            'jetons_provisionnes': self.provision, 'jetons_factures': self.provision,
            'etapes': 1, 'succes': False, 'duree': 0})

    def terminer(self, usage=None, succes=False):
        with self.verrou:
            if self.termine:
                return
            self.termine = True
            usage = dict(usage or {})
            total = usage.get('total')
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                entree, sortie = usage.get('entree'), usage.get('sortie')
                total = entree + sortie if all(isinstance(v, int) and not isinstance(v, bool)
                                               and v >= 0 for v in (entree, sortie)) else None
            # Les SDK construisent aussi des compteurs zéro si le serveur n'en fournit pas.
            connu = total is not None and total > 0 and usage.get('usage_connu') is not False
            facture = total if connu else self.provision
            if connu:
                try:
                    self.reservation.rapprocher(self.provision, total)
                except Exception as exc:
                    logger.warning('rapprochement budget indisponible type=%s', type(exc).__name__)
                    facture = self.provision
            elif succes:
                logger.warning(
                    'usage fournisseur absent apres succes phase=%s fournisseur=%s provision=%d duree=%.2f',
                    self.phase, self.fournisseur, self.provision, time.monotonic() - self.depart,
                )
                try:
                    self.reservation.noter_usage_inconnu()
                except Exception as exc:
                    logger.warning('mesure inconnue non enregistree type=%s', type(exc).__name__)
            self.trace.remplacer(self.index, {**usage, 'phase': self.phase, 'fournisseur': self.fournisseur,
                'usage_connu': connu, 'total': total if connu else None,
                'jetons_provisionnes': self.provision, 'jetons_factures': facture,
                'etapes': 1, 'succes': succes, 'duree': time.monotonic() - self.depart})


@contextmanager
def appel_fournisseur(*, phase, fournisseur, jetons_entree, max_sortie=4096):
    appel = AppelFournisseur(phase, fournisseur, jetons_entree, max_sortie)
    try:
        yield appel
    finally:
        appel.terminer()


def _usage_sdk(usage):
    return {'entree': usage.input_tokens, 'sortie': usage.output_tokens,
            'total': usage.total_tokens, 'cache': usage.cache_read_tokens}


def _operation_db(fonction, *args):
    """Les threads du SDK ne conservent pas de connexion au-delà de l'opération."""
    try:
        return fonction(*args)
    finally:
        connections.close_all()


def modele_borne(modele, phase='planning'):
    """Chaque tentative de fallback possède sa propre provision."""
    from pydantic_ai.models.fallback import FallbackModel
    if isinstance(modele, ModeleBorne):
        modele.phase = phase
        return modele
    if isinstance(modele, FallbackModel):
        return FallbackModel(*(modele_borne(m, phase) for m in modele.models),
                             fallback_on=modele._fallback_on)
    return ModeleBorne(modele, phase)


from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic import TypeAdapter
from pydantic_ai.models import ModelRequestParameters

_PARAMETRES = TypeAdapter(ModelRequestParameters)


class ModeleBorne(WrapperModel):
    def __init__(self, modele, phase):
        super().__init__(modele)
        self.phase = phase

    async def _ouvrir(self, messages, reglages, parametres):
        octets = len(ModelMessagesTypeAdapter.dump_json(messages)) + len(
            _PARAMETRES.dump_json(parametres))
        entree = (octets + 2) // 3 + 2048
        reglages = dict(reglages or {})
        appel = await asyncio.to_thread(_operation_db, AppelFournisseur, self.phase, self.system,
                                        entree, reglages.get('max_tokens', 4096))
        reglages['max_tokens'] = appel.max_sortie
        appel.delai = min(float(reglages.get('timeout') or 60), appel.delai)
        reglages['timeout'] = appel.delai
        # Google refuse un en-tête HTTP inférieur à 10 s. Le délai local
        # reste celui du tour, même quand son transport exige ce minimum.
        if self.system in ('google-gla', 'google-vertex'):
            reglages['timeout'] = max(10.0, appel.delai)
        return appel, reglages

    async def request(self, messages, model_settings, model_request_parameters):
        appel, reglages = await self._ouvrir(messages, model_settings, model_request_parameters)
        try:
            reponse = await asyncio.wait_for(
                self.wrapped.request(messages, reglages, model_request_parameters),
                timeout=appel.delai)
            await asyncio.to_thread(_operation_db, appel.terminer, _usage_sdk(reponse.usage), True)
            return reponse
        finally:
            await asyncio.to_thread(_operation_db, appel.terminer)

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
        appel, reglages = await self._ouvrir(messages, model_settings, model_request_parameters)
        try:
            with anyio.fail_after(appel.delai):
                async with self.wrapped.request_stream(messages, reglages,
                        model_request_parameters, run_context) as reponse:
                    yield reponse
                    await asyncio.to_thread(_operation_db, appel.terminer, _usage_sdk(reponse.usage()), True)
        except TimeoutError as exc:
            # Sur Python 3.10, AnyIO et asyncio n'utilisent pas la même classe.
            raise asyncio.TimeoutError from exc
        finally:
            await asyncio.to_thread(_operation_db, appel.terminer)
# redeploy
