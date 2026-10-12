"""
L'import d'un document, inscrit au registre du tour.

Le processeur de documents cree les blocs HORS de la boucle d'outils: quand
l'utilisateur envoie son horaire, l'import est fait par le systeme avant meme
qu'AGIR ne tourne. Jusqu'au 2026-09-01, seul AGIR en etait informe (par le
message enrichi); DIRE recevait « REGISTRE DU TOUR: VIDE. Tu n'as rien
accompli » et, docile, proposait a l'utilisateur d'envoyer l'horaire qu'il
venait d'importer. Deux tours reels, deux recaps faux, le jour de la
soumission 1.1.

Ici l'import devient une entree du registre, ecrite par du code depuis la
base: l'utilisateur lit un recap rendu par le systeme, et DIRE a une
reference a citer.

- Document envoye CE tour: `import_document`, une MUTATION. Le bloc factuel
  affiche le recap (cours ajoutes, blocs a confirmer, entrees ignorees).
- Import recent sans piece jointe (« c'est bon ? » au tour suivant):
  `import_recent`, une LECTURE. DIRE connait les blocs et peut repondre, le
  bloc factuel ne rejoue pas l'import a chaque tour pendant vingt minutes.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, UploadedDocument
from services.agent.tools.base import ToolResult

from .registre import Action, Registre

# Meme fenetre que v1 (_recent_import_context): au-dela, un import n'est plus
# « ce que l'utilisateur vient de faire ».
FENETRE_IMPORT_RECENT = timedelta(minutes=20)
JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _hm(valeur) -> str:
    if valeur is None:
        return "?"
    if hasattr(valeur, "strftime"):
        return valeur.strftime("%H:%M")
    return str(valeur)[:5]


def document_du_tour(user, attachment) -> Optional[UploadedDocument]:
    """Le document dont l'import compte pour ce tour, s'il est analyse."""
    if attachment is not None:
        return UploadedDocument.objects.filter(
            pk=attachment.pk, user=user, processed=True, processing_error__isnull=True
        ).first()
    limite = timezone.now() - FENETRE_IMPORT_RECENT
    doc = (UploadedDocument.objects
           .filter(user=user, uploaded_at__gte=limite)
           .order_by("-uploaded_at")
           .first())
    if doc is None or not doc.processed:
        return None
    return doc


def resume_import(doc: UploadedDocument) -> dict:
    """Ce que l'import a REELLEMENT produit, lu en base, jamais deduit."""
    blocs = list(RecurringBlock.objects
                 .filter(source_document=doc, user=doc.user)
                 .order_by("day_of_week", "start_time"))
    en_attente = (RecurringBlock.all_objects
                  .filter(source_document=doc, user=doc.user, status=RecurringBlock.STATUS_PENDING)
                  .count())
    dates = list(ScheduledBlock.objects
                 .filter(source_document=doc, user=doc.user, task__user=doc.user)
                 .select_related("task")
                 .order_by("date", "start_time"))

    # Les entrees lues dans le document mais absentes de la base ont ete
    # ecartees par le processeur (chevauchement avec un bloc en place). Le
    # processeur ne les persiste pas: on les retrouve par difference, ce qui
    # est la seule facon de dire a l'utilisateur pourquoi il lui manque un
    # cours. Vecu le 2026-09-01: 5 cours lus, 4 crees, et un recap muet.
    extraits = (doc.extracted_data or {}).get("courses") or []
    connus = {b.title.strip().lower()
              for b in RecurringBlock.all_objects.filter(source_document=doc, user=doc.user)}
    ignores = []
    for cours in extraits:
        if not isinstance(cours, dict):
            continue
        nom = (cours.get("name") or "").strip()
        if not nom or nom.lower() in connus:
            continue
        ignores.append({
            "titre": nom,
            "jour": str(cours.get("day") or "?"),
            "debut": str(cours.get("start_time") or "?"),
            "fin": str(cours.get("end_time") or "?"),
            "reason": "unknown",
        })
    data = doc.extracted_data or {}
    if data.get('_import_receipt_version'):
        ignores = data.get('_ignored_entries') or []
    precisions = data.get('_pending_entries') or []

    return {
        "document_id": doc.pk,
        "fichier": doc.file_name,
        "blocs": [{
            "id": b.id,
            "titre": b.title,
            # Pour le recap rendu (rendu.py): « 3 cours » ou « 2 quarts »
            # selon la nature du bloc. message_import ne le lit pas.
            "type": b.block_type,
            "jour": JOURS[b.day_of_week] if 0 <= b.day_of_week < 7 else "?",
            "debut": _hm(b.start_time),
            "fin": _hm(b.end_time),
            "debute_le": b.start_date.isoformat() if b.start_date else None,
            "finit_le": b.end_date.isoformat() if b.end_date else None,
        } for b in blocs],
        "en_attente": en_attente + len(precisions),
        "blocs_a_confirmer": en_attente,
        "precisions": precisions,
        "dates": [{
            "id": sb.pk,
            "titre": sb.task.title,
            "date": sb.date.isoformat(),
            "debut": _hm(sb.start_time),
            "fin": _hm(sb.end_time),
        } for sb in dates],
        "ignores": ignores,
    }


def resume_precision(user, document_id, source, index):
    """Relit une précision et ses cibles exactes, pour vérifier son reçu."""
    doc = UploadedDocument.objects.filter(pk=document_id, user=user, processed=True,
                                         processing_error__isnull=True).first()
    if doc is None or source not in ('courses', 'shifts', 'events'):
        return None
    data = doc.extracted_data or {}
    entries = data.get(source) or []
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(entries):
        return None
    pending = next((p for p in data.get('_pending_entries', [])
                    if p.get('source') == source and p.get('index') == index), None)
    ledger = (data.get('_resolved_entries') or {}).get(f'{source}:{index}')
    if pending is None and not ledger:
        return None
    targets = []
    for target in (ledger or {}).get('targets', []):
        if target.get('model') == 'RecurringBlock':
            block = RecurringBlock.all_objects.filter(pk=target.get('id'), user=user,
                                                     source_document=doc).first()
            if block is None:
                return None
            targets.append({'model': 'RecurringBlock', 'id': block.pk, 'title': block.title,
                'day_of_week': block.day_of_week, 'start_time': _hm(block.start_time),
                'end_time': _hm(block.end_time), 'status': block.status, 'active': block.active,
                'source_document_id': doc.pk})
        elif target.get('model') == 'ScheduledBlock':
            block = ScheduledBlock.objects.filter(pk=target.get('id'), user=user,
                task__user=user, source_document=doc).select_related('task').first()
            if block is None:
                return None
            targets.append({'model': 'ScheduledBlock', 'id': block.pk, 'title': block.task.title,
                'date': block.date.isoformat(), 'start_time': _hm(block.start_time),
                'end_time': _hm(block.end_time), 'locked': block.locked, 'source_document_id': doc.pk})
        else:
            return None
    if pending is None and not targets:
        return None
    return {'document_id': doc.pk, 'source': source, 'index': index,
            'entry': entries[index], 'pending': pending is not None,
            'missing_fields': (pending or {}).get('missing_fields', []), 'targets': targets}


def _pluriel(n: int, mot: str, participe: str) -> str:
    if n <= 1:
        return f"{n} {mot} {participe}"
    return f"{n} {mot}s {participe}s"


def message_import(resume: dict) -> str:
    """Le recap, rendu par du code. C'est ce que l'utilisateur lit."""
    blocs, dates = resume["blocs"], resume["dates"]
    total = len(blocs) + len(dates)
    if not total and not resume["en_attente"] and not resume['ignores']:
        return (f"Document « {resume['fichier']} » analysé : aucun cours, quart ou "
                "événement exploitable n'en est sorti.")
    lignes = ([f"Horaire importé depuis « {resume['fichier']} » : "
               f"{_pluriel(total, 'entrée', 'ajoutée')}"] if total else
              [f"Document « {resume['fichier']} » analysé : aucune nouvelle entrée ajoutée."])
    for b in blocs:
        lignes.append(f"  - {b['titre']} : {b['jour']} {b['debut']}-{b['fin']}")
    for d in dates:
        lignes.append(f"  - {d['titre']} : {d['date']} {d['debut']}-{d['fin']}")
    a_confirmer = resume.get('blocs_a_confirmer', resume['en_attente'])
    if a_confirmer:
        lignes.append(f"  - {_pluriel(a_confirmer, 'bloc', 'lu')} avec un doute, "
                      "à confirmer dans le planning")
    for precision in resume.get('precisions', []):
        lignes.append(f"  - {message_precision(precision)}")
    for i in resume["ignores"]:
        lignes.append(f"  - Non ajouté : {i['titre']} ({i['jour']} {i['debut']}-{i['fin']}), "
                      + raison_ignoree(i.get('reason')))
    return "\n".join(lignes)


def raison_ignoree(reason):
    return {'duplicate': 'ce créneau existe déjà',
            'overlap': 'chevauche un bloc déjà en place'}.get(
                reason, "aucune création vérifiée pour cette entrée")


def message_precision(precision):
    entry = precision.get('entry') or {}
    title = entry.get('title') or entry.get('name') or entry.get('role') or 'Cette entrée'
    labels = {'date': 'la date', 'day': 'le jour', 'start_time': "l'heure de début",
              'end_time': "l'heure de fin"}
    champs = ', '.join(labels.get(k, k) for k in precision.get('missing_fields', []))
    return f"{title} : précise {champs} pour l'ajouter au planning."


def inscrire_import(registre: Registre, user, attachment) -> Optional[Action]:
    """Inscrit l'import au registre; rend l'entree, ou None s'il n'y a rien.

    Sans piece jointe, un document qui n'a rien produit ne s'annonce pas (un
    « bonjour » au tour suivant n'a pas a en entendre parler). Avec la piece
    jointe, il s'inscrit en LECTURE: DIRE doit pouvoir dire honnetement que
    rien d'exploitable n'en est sorti, sans qu'un bloc factuel n'annonce un
    import qui n'a rien ajoute.
    """
    doc = document_du_tour(user, attachment)
    if doc is None:
        return None
    resume = resume_import(doc)
    vide = not resume["blocs"] and not resume["dates"] and not resume["en_attente"] and not resume['ignores']
    if vide and attachment is None:
        return None
    outil = "import_document" if attachment is not None and not vide else "import_recent"
    return registre.ajouter(
        outil, {"document": doc.file_name, "document_id": doc.pk},
        ToolResult(success=True, message=message_import(resume), data=resume))
