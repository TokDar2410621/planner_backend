"""
Relire ce qu'on vient d'ecrire, et comparer par du code.

Aucun outil ne relit la base apres ecriture: leur `data` est construit depuis
l'instance en memoire. C'est exactement pourquoi cette etape existe.

Les chemins de date ci-dessous viennent du code REEL, outil par outil. Une
seule erreur de chemin rendrait le detecteur muet en production tout en
laissant ses tests au vert.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from services.agent_v2.registre import Postcondition, Registre

logger = logging.getLogger(__name__)

LECTURES: dict[str, str] = {
    "create_block": "list_blocks",
    "update_block": "list_blocks",
    "delete_block": "list_blocks",
    "clear_all_blocks": "list_blocks",
    "skip_block_occurrence": "get_today_schedule",
    "replace_block_occurrence": "get_today_schedule",
    "restore_block_occurrence": "get_today_schedule",
    "create_task": "list_tasks",
    "update_task": "list_tasks",
    "delete_task": "list_tasks",
    "complete_task": "list_tasks",
    "schedule_task_at": "get_week_schedule",
    "cancel_scheduled_block": "get_week_schedule",
    "optimize_week": "get_week_schedule",
    "organize_day": "get_today_schedule",
    "update_preferences": "get_preferences",
    "create_goal": "list_goals",
    "update_goal": "list_goals",
    "import_document": "list_blocks",
    "resolve_document_entry": "process_uploaded_document",
}

# Ou trouver la date obtenue, par outil. Chemins verifies dans le code v1.
CHEMINS_DATE: dict[str, tuple] = {
    "schedule_task_at": ("scheduled_block", "date"),
    "skip_block_occurrence": ("date",),
    # Les heures et le titre se lisent a cote de la date (chemin de longueur
    # 2): au premier niveau, l'ecart « cree dans le passe » sortirait sans
    # heures ni titre.
    "replace_block_occurrence": ("scheduled_block", "date"),
    "restore_block_occurrence": ("date",),
    "organize_day": ("date",),
    "optimize_week": ("start_date",),
}


def _lire(donnees: dict, chemin: tuple):
    valeur = donnees
    for cle in chemin:
        if not isinstance(valeur, dict):
            return None
        valeur = valeur.get(cle)
    return valeur


def _egal(champ, a, b):
    if champ == "deadline" and a and b:
        try:
            return datetime.fromisoformat(str(a)) == datetime.fromisoformat(str(b))
        except ValueError:
            return False
    return a == b


def _preuve(entite, cible, attendu, observe):
    conforme = bool(attendu and observe is not None
                    and all(_egal(k, observe.get(k), v) for k, v in attendu.items()))
    return Postcondition(entite, cible, attendu, observe, conforme)


def _attendu(resultat, parametres, champs):
    """Les valeurs déclarées ET celles demandées, sans interprétation de prose."""
    attendu = {k: v for k, v in resultat.items() if k in champs}
    for k in champs:
        if k in parametres and parametres[k] is not None:
            valeur = parametres[k]
            if isinstance(valeur, str):
                valeur = valeur.strip()
            if k in ("start_time", "end_time"):
                valeur = str(valeur)[:5]
            if k in ("description", "location") and valeur == "":
                valeur = None
            if k == "day_of_week":
                from services.agent.tools.blocks import normaliser_jours
                jours = normaliser_jours([valeur])
                if len(jours) != 1:
                    raise ValueError("jour non vérifiable")
                valeur = jours[0]
            if k in attendu and not _egal(k, attendu[k], valeur):
                raise ValueError("résultat différent de la demande")
            attendu[k] = valeur
    return attendu


def _objets(user, action, modele, serialiser, entrees, champs, identifiant=None):
    preuves = []
    for entree in entrees:
        if not isinstance(entree, dict) or entree.get("id") is None:
            return []
        cible = {"id": entree["id"], "user_id": user.pk}
        cibles = modele.objects.filter(user=user, pk=entree["id"])
        if modele.__name__ == "RecurringBlock":
            cibles = cibles.filter(active=True)
        objet = cibles.first()
        parametres = dict(action.parametres)
        if modele.__name__ == "Task" and parametres.get("deadline"):
            from services.agent.tools.tasks import _deadline_iso, _parse_deadline
            parametres["deadline"] = _deadline_iso(_parse_deadline(parametres["deadline"]))
        # Une création dédoublonnée constate l'objet existant ; les autres
        # paramètres n'ont pas été appliqués et ne doivent pas l'être au reçu.
        if action.outil == "create_task" and action.donnees.get("deja_presente"):
            parametres = {}
        attendu = _attendu(entree, parametres, champs)
        attendu["id"] = action.parametres.get(identifiant, entree["id"]) if identifiant else entree["id"]
        observe = serialiser(objet) if objet is not None else None
        preuves.append(_preuve(modele.__name__, cible, attendu, observe))
    return preuves


def _placement(user, action):
    from core.models import ScheduledBlock
    donnees = action.donnees.get("scheduled_block") or {}
    if not all(k in donnees for k in ("id", "date", "title", "start_time", "end_time")):
        return []
    sb = ScheduledBlock.objects.filter(user=user, pk=donnees["id"]).select_related("task").first()
    parametres = dict(action.parametres)
    if action.outil == "replace_block_occurrence":
        parametres["title"] = parametres.get("replacement_title", donnees["title"])
    attendu = _attendu(donnees, parametres,
                      {"id", "date", "title", "start_time", "end_time", "overnight"})
    observe = None
    if sb is not None and sb.task is not None and sb.task.user_id == user.pk:
        observe = {"id": sb.pk, "date": sb.date.isoformat(), "title": sb.task.title,
                   "start_time": sb.start_time.strftime("%H:%M"),
                   "end_time": sb.end_time.strftime("%H:%M"), "overnight": False}
        if donnees.get("overnight"):
            # Une nuit est conservée en deux lignes. Les deux sont nécessaires.
            queue = ScheduledBlock.objects.filter(
                user=user, task_id=sb.task_id, date=sb.date + timedelta(days=1),
                start_time="00:00", end_time=attendu["end_time"]).first()
            if queue is not None and sb.end_time.strftime("%H:%M") == "23:59":
                observe["end_time"] = queue.end_time.strftime("%H:%M")
                observe["overnight"] = True
    return [_preuve("ScheduledBlock", {"id": donnees["id"], "user_id": user.pk}, attendu, observe)]


def _exception(user, action):
    from core.models import RecurringBlock, RecurringBlockException
    donnees = action.donnees
    description = donnees.get("remplace") or donnees
    jour = date.fromisoformat(donnees["date"])
    blocs = RecurringBlock.objects.filter(user=user, active=True, title=description["title"],
                                          day_of_week=jour.weekday())
    for cle in ("start_time", "end_time", "block_type"):
        if description.get(cle) is not None:
            blocs = blocs.filter(**{cle: description[cle]})
    candidats = [b for b in blocs if (not b.start_date or b.start_date <= jour)
                 and (not b.end_date or jour <= b.end_date)]
    if len(candidats) != 1:
        return []
    bloc = candidats[0]
    existe = RecurringBlockException.objects.filter(user=user, recurring_block=bloc, date=jour).exists()
    attendu = {"date": action.parametres.get("date", donnees["date"]),
               "annulee": action.outil != "restore_block_occurrence"}
    observe = {"date": jour.isoformat(), "annulee": existe}
    return [_preuve("RecurringBlockException", {"block_id": bloc.pk, "date": jour.isoformat(),
                                                "user_id": user.pk}, attendu, observe)]


def _plan(user, action):
    """Vérifie chaque placement annoncé à sa date et avec son identité."""
    from core.models import RecurringBlockOverride
    from services.scheduling.day_view import effective_day_entries
    jours = action.donnees.get("days") if action.outil == "optimize_week" else [action.donnees]
    debut = action.parametres.get("start_date") or action.parametres.get("date")
    if len(jours or []) != (7 if action.outil == "optimize_week" else 1):
        return []
    if debut and jours and jours[0].get("date") != debut:
        return []
    preuves = []
    for offset, jour in enumerate(jours or []):
        jour_date = date.fromisoformat(jour["date"])
        if jour_date != date.fromisoformat(jours[0]["date"]) + timedelta(days=offset):
            return []
        effectives = effective_day_entries(user, jour_date)
        for place in [*(jour.get("placed") or []), *(jour.get("moved") or [])]:
            identifiant = place.get("block_id")
            if identifiant is None:
                return []
            correspondances = [b for b in effectives if b.get("type") == "recurring" and b.get("id") == identifiant]
            attendu = {k: place[k] for k in ("title", "start_time", "end_time")}
            observe = correspondances[0] if len(correspondances) == 1 else None
            preuves.append(_preuve("Occurrence", {"block_id": identifiant, "date": jour_date.isoformat(),
                                                  "user_id": user.pk}, attendu, observe))
            override = RecurringBlockOverride.objects.filter(
                user=user, recurring_block__user=user, recurring_block_id=identifiant, date=jour_date,
            ).first()
            preuves.append(_preuve(
                "RecurringBlockOverride", {"block_id": identifiant, "date": jour_date.isoformat(),
                                           "user_id": user.pk},
                {k: place[k] for k in ("start_time", "end_time")},
                {"start_time": override.start_time.strftime("%H:%M"),
                 "end_time": override.end_time.strftime("%H:%M")} if override else None,
            ))
    return preuves


def _postconditions(user, action):
    from core.models import Goal, RecurringBlock, ScheduledBlock, Task, UploadedDocument, UserProfile
    from services.agent.tools.blocks import _block_to_dict
    from services.agent.tools.goals import _goal_to_dict
    from services.agent.tools.tasks import _task_to_dict

    outil, d, p = action.outil, action.donnees, action.parametres
    sans_mutation = False
    if outil == 'resolve_document_entry':
        from services.agent_v2.importation import resume_precision
        observed = resume_precision(user, p.get('document_id'), p.get('source'), p.get('index'))
        return [_preuve('UploadedDocument', {'id': p.get('document_id'), 'user_id': user.pk,
            'source': p.get('source'), 'index': p.get('index')},
            d.get('resolution') or {}, observed)], bool(d.get('already_resolved'))
    if outil in ("create_block", "update_block"):
        entrees = d.get("created", []) if outil == "create_block" else [d.get("block", {})]
        preuves = _objets(user, action, RecurringBlock, _block_to_dict, entrees,
                         {"title", "block_type", "day_of_week", "start_time", "end_time", "flexibility",
                          "location", "is_night_shift", "start_date", "end_date"}, "block_id")
        return preuves, False
    if outil in ("create_task", "update_task", "complete_task"):
        preuves = _objets(user, action, Task, _task_to_dict, [d.get("task", {})],
                         {"title", "description", "task_type", "priority", "deadline",
                          "estimated_duration_minutes", "completed", "place"}, "task_id")
        if outil == "complete_task" and preuves:
            preuve = preuves[0]
            preuves[0] = _preuve(preuve.entite, preuve.cible,
                                 {**preuve.attendu, "completed": True}, preuve.observe)
        return preuves, bool(d.get("deja_presente"))
    if outil in ("create_goal", "update_goal"):
        return _objets(user, action, Goal, _goal_to_dict, [d.get("goal", {})],
                       {"title", "description", "goal_type", "deadline", "progress", "status"}, "goal_id"), False
    if outil in ("schedule_task_at", "replace_block_occurrence"):
        preuves = _placement(user, action)
        if outil == "replace_block_occurrence":
            autres = _exception(user, action)
            preuves = preuves + autres if preuves and autres else []
        return preuves, False
    if outil in ("skip_block_occurrence", "restore_block_occurrence"):
        return _exception(user, action), d.get("restored") is False
    if outil == "delete_task":
        identifiant = d.get("deleted_id")
        if identifiant is None or p.get("task_id") != identifiant:
            return [], False
        existe = Task.objects.filter(user=user, pk=identifiant).exists()
        return [_preuve("Task", {"id": identifiant, "user_id": user.pk}, {"existe": False}, {"existe": existe})], False
    if outil == "delete_block":
        identifiant = d.get("deleted_id")
        if identifiant is None or p.get("block_id") != identifiant:
            return [], False
        bloc = RecurringBlock.all_objects.filter(user=user, pk=identifiant).first()
        return [_preuve("RecurringBlock", {"id": identifiant, "user_id": user.pk}, {"active": False},
                        {"active": bloc.active} if bloc is not None else None)], False
    if outil == "clear_all_blocks":
        reste = RecurringBlock.objects.filter(user=user, active=True).count()
        return [_preuve("RecurringBlock", {"user_id": user.pk}, {"actifs": 0}, {"actifs": reste})], False
    if outil == "cancel_scheduled_block":
        preuves = []
        jour_cible = date.fromisoformat(p["date"])
        entrees = d.get("cancelled") or []
        if not entrees or entrees[0].get("date") != jour_cible.isoformat():
            return [], False
        for entree in entrees:
            if not all(k in entree for k in ("title", "date", "start_time", "end_time")):
                return [], False
            if date.fromisoformat(entree["date"]) not in (jour_cible, jour_cible + timedelta(days=1)):
                return [], False
            cible = {"user_id": user.pk, **entree}
            existe = ScheduledBlock.objects.filter(user=user, task__title=entree["title"], date=entree["date"],
                                                   start_time=entree["start_time"], end_time=entree["end_time"]).exists()
            preuves.append(_preuve("ScheduledBlock", cible, {"existe": False}, {"existe": existe}))
        return preuves, False
    if outil == "update_preferences":
        champs = d.get("updated_fields") or []
        profil = UserProfile.objects.filter(user=user).values(*champs).first() if champs else None
        attendu = {k: p[k] for k in champs if k in p}
        return ([_preuve("UserProfile", {"user_id": user.pk}, attendu, profil)] if champs else []), not champs
    if outil in ("organize_day", "optimize_week"):
        if d.get("applied") is False:
            # Le plan n'est pas une mutation : son affichage ne confirme rien en base.
            return [], True
        return _plan(user, action), d.get("moved_count") == 0
    if outil == "import_document":
        from services.agent_v2.importation import resume_import
        identifiant = p.get("document_id")
        if identifiant is None:
            return [], False
        doc = UploadedDocument.objects.filter(user=user, pk=identifiant, processed=True).first()
        preuves = [_preuve("UploadedDocument", {"id": identifiant, "user_id": user.pk}, d,
                           resume_import(doc) if doc is not None else None)]
        for bloc in d.get("blocs") or []:
            actif = RecurringBlock.objects.filter(user=user, source_document_id=identifiant,
                                                   pk=bloc.get("id"), active=True).exists()
            preuves.append(_preuve("RecurringBlock", {"id": bloc.get("id"), "user_id": user.pk},
                                   {"active": True}, {"active": actif}))
        return preuves, False
    return [], sans_mutation


def reconcilier(user: User, registre: Registre) -> dict:
    """Relit les cibles exactes, compare, puis publie le statut du reçu."""
    etat = {}
    for action in registre.mutations():
        if not action.succes:
            continue
        preuves, sans_mutation = [], False
        try:
            preuves, sans_mutation = _postconditions(user, action)
        except Exception as exc:  # noqa: BLE001 - une relecture ratée ne confirme rien
            logger.warning("Reconciliation %s impossible (%s)", action.outil, type(exc).__name__)
        registre.verifier(action.id, user.pk, preuves, sans_mutation=sans_mutation)
        etat[action.id] = registre.par_id(action.id).recu
    return etat


def _est_passe(date_txt, heure_fin_txt) -> bool:
    """La chose placee est-elle deja derriere nous ?

    On compare la FIN, pas la date seule: le cas reel du 2026-08-26 est un
    creneau 09:00-12:00 cale le jour meme a 23h30. Sans l'heure, on l'aurait
    declare valide.
    """
    from datetime import datetime, time as _time

    try:
        jour = datetime.strptime(str(date_txt)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    try:
        fin = datetime.strptime(str(heure_fin_txt)[:5], "%H:%M").time()
    except (TypeError, ValueError):
        fin = _time(23, 59)
    maintenant = timezone.localtime()
    return timezone.make_aware(
        datetime.combine(jour, fin), maintenant.tzinfo) < maintenant


def detecter_ecarts(registre: Registre) -> None:
    """Compare l'intention machine au resultat reel."""
    for action in registre.actions:
        if not action.etat_verifie or not action.est_mutation:
            continue

        chemin = CHEMINS_DATE.get(action.outil)
        if chemin:
            # Une seance placee dans le PASSE est une mutation reelle et
            # inutile: la validation des references ne la voit pas, puisque
            # l'action a bien eu lieu. Observe le 2026-08-26 sur un tour reel:
            # 4 revisions sur 7 calees derriere l'heure courante, toutes
            # annoncees comme calees.
            obtenue_brute = _lire(action.donnees, chemin)
            fin = _lire(action.donnees, chemin[:-1] + ("end_time",)) if len(chemin) > 1 \
                else action.donnees.get("end_time")
            # Ce que rendu.py affiche a l'utilisateur, lu aux memes endroits
            # que la date: la description ci-dessous reste pour le modele.
            debut = _lire(action.donnees, chemin[:-1] + ("start_time",)) if len(chemin) > 1 \
                else action.donnees.get("start_time")
            titre = _lire(action.donnees, chemin[:-1] + ("title",)) if len(chemin) > 1 \
                else action.donnees.get("title")
            if obtenue_brute and _est_passe(obtenue_brute, fin):
                # Formulation deliberement explicite sur le fait que la chose
                # EXISTE. Une premiere version disait « place dans le passe,
                # donc inutilisable » et DIRE l'a rendue par « je l'ai laisse
                # de cote » (tour reel du 2026-08-26), alors que le bloc etait
                # bel et bien en base. Un ecart mal formule redevient un
                # mensonge, par le seul chemin qui reste ouvert.
                registre.ajouter_ecart(
                    action.id,
                    f"CREE mais dans le passe ({obtenue_brute} {fin or ''}"
                    .rstrip() + "): il existe et n'a PAS ete annule, il est "
                    "seulement inutilisable tel quel et doit etre replace",
                    genre="passe",
                    donnees={"date": str(obtenue_brute)[:10], "debut": debut,
                             "fin": fin, "titre": titre})
            demandee = action.parametres.get("date") or action.parametres.get("start_date")
            obtenue = _lire(action.donnees, chemin)
            if demandee and obtenue and str(demandee) != str(obtenue):
                registre.ajouter_ecart(
                    action.id, f"date demandee {demandee}, date obtenue {obtenue}",
                    genre="date_differente",
                    donnees={"demandee": str(demandee), "obtenue": str(obtenue),
                             "titre": titre})

        # Succes SANS mutation reelle: verifie dans le code v1, quatre cas.
        if action.outil == "create_task" and action.donnees.get("deja_presente"):
            tache = action.donnees.get("task") or {}
            registre.ajouter_ecart(
                action.id, "tache deja presente, rien n'a ete cree",
                genre="tache_existante",
                donnees={"titre": tache.get("title") if isinstance(tache, dict) else None})
        if action.outil in ("optimize_week", "organize_day") and action.donnees.get("applied") is False:
            registre.ajouter_ecart(
                action.id, "plan seulement propose, rien n'a ete applique",
                genre="plan_propose",
                donnees={"date": action.donnees.get("date") or action.donnees.get("start_date")})
        if action.outil == "update_preferences" and not action.donnees.get("updated_fields"):
            registre.ajouter_ecart(action.id, "aucune preference n'a change",
                                   genre="preferences_inchangees")
        if action.outil == "restore_block_occurrence" and action.donnees.get("restored") is False:
            registre.ajouter_ecart(
                action.id, "aucune occurrence sautee a restaurer",
                genre="rien_a_restaurer",
                donnees={"date": action.donnees.get("date"),
                         "titre": action.donnees.get("title")})
