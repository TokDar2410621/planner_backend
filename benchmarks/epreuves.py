"""
Les épreuves du banc. Chacune rend une Note (points obtenus / possibles).

Règle de notation: on lit l'ÉTAT DE LA BASE, jamais le récit de l'agent. Les
seules exceptions sont l'épreuve « vérité d'action » (qui compare justement le
récit aux appels d'outils réels) et « ton » (juge LLM sur grille).
"""
from __future__ import annotations

from datetime import date, timedelta

from django.utils import timezone

from .harness import Monde, Note, Pilote, hhmm, monde_neuf


def _semaine_type(m: Monde) -> None:
    """Planning de départ commun: 3 cours, un quart de nuit, du sommeil."""
    m.bloc("Mathématiques", 0, "09:00", "12:00")
    m.bloc("Physique", 2, "09:00", "12:00")
    m.bloc("Labo de chimie", 4, "13:00", "16:00")
    m.bloc("Quart entrepôt", 3, "23:00", "07:00", btype="work")
    for dow in (0, 1, 2, 6):
        m.bloc("Sommeil", dow, "23:30", "07:00", btype="sleep")


# ------------------------------------------------------- 1. temporel (20 pts)

def epreuve_temporel(p: Pilote) -> Note:
    n = Note("Compréhension temporelle", 0, 0)
    maintenant = timezone.localtime()

    # a) « dans 25 minutes »: doit être placé aujourd'hui, à l'heure demandée.
    m = monde_neuf("temps-a")
    _semaine_type(m)
    t = p.envoyer(m.user, "Planifie une pause café de 15 minutes dans 25 minutes.")
    n.tours.append(t)
    vise = (maintenant + timedelta(minutes=25)).time()
    places = m.places(maintenant.date())
    trouve = [s for s in places if abs(
        (s.start_time.hour * 60 + s.start_time.minute) - (vise.hour * 60 + vise.minute)) <= 10]
    n.point(bool(trouve), f"« dans 25 min » placé vers {vise:%H:%M} (trouvé: "
                          f"{[hhmm(s.start_time) for s in places] or 'rien'})", 4)

    # b) Jour et date incohérents: doit demander, PAS inventer.
    m = monde_neuf("temps-b")
    _semaine_type(m)
    faux_lundi = _prochain_mercredi(maintenant.date())
    t = p.envoyer(m.user, f"Mets lundi {faux_lundi:%d %B} à 13h une rencontre d'équipe.")
    n.tours.append(t)
    cree = m.places(faux_lundi)
    n.point(not cree or "?" in t.reponse,
            "incohérence jour/date signalée au lieu d'être devinée en silence", 4)

    # c) Date passée: doit refuser ou signaler, jamais planifier dans le passé.
    m = monde_neuf("temps-c")
    _semaine_type(m)
    hier = maintenant.date() - timedelta(days=1)
    t = p.envoyer(m.user, f"Ajoute un rendez-vous dentiste le {hier:%d %B} à 10h.")
    n.tours.append(t)
    n.point(not m.places(hier), "refuse de planifier dans le passé", 4)

    # d) Quart overnight: end < start conservé, pas inversé.
    m = monde_neuf("temps-d")
    _semaine_type(m)
    t = p.envoyer(m.user, "J'ai un nouveau quart au dépanneur le samedi de 22h à 6h du matin.")
    n.tours.append(t)
    quarts = [b for b in m.blocs("dépanneur")] or [b for b in m.blocs("quart")
                                                   if b.day_of_week == 5]
    ok = any(b.day_of_week == 5 and hhmm(b.start_time) == "22:00"
             and hhmm(b.end_time) == "06:00" for b in quarts)
    n.point(ok, "quart overnight samedi 22:00-06:00 avec end < start", 4)

    # e) Lecture d'un jour précis: aucune écriture parasite.
    m = monde_neuf("temps-e")
    _semaine_type(m)
    avant = len(m.blocs()) + len(m.places())
    t = p.envoyer(m.user, "J'ai quoi mercredi prochain ?")
    n.tours.append(t)
    apres = len(m.blocs()) + len(m.places())
    n.point(avant == apres, "une question de lecture n'écrit rien", 4)

    # L'epreuve pesait 5 items a 4 points: un seul echec coutait 4 points et
    # l'amplitude entre passages atteignait 8. Des items plus nombreux et plus
    # legers rendent la note representative de la competence, pas du hasard.
    t = p.envoyer(m.user, "j'ai quoi le 30 fevrier ?")
    n.tours.append(t)
    rep = (t.reponse or "").lower()
    n.point(("30" in rep and ("existe" in rep or "valide" in rep or "erreur" in rep))
            or "n'existe pas" in rep, "une date impossible est signalee", 2)

    t = p.envoyer(m.user, "mon quart de nuit finit a quelle heure ?")
    n.tours.append(t)
    n.point("07" in (t.reponse or "") or "7 h" in (t.reponse or ""),
            "l'heure de fin d'un overnight est lue correctement", 2)

    t = p.envoyer(m.user, "c'est quel jour de la semaine aujourd'hui ?")
    n.tours.append(t)
    jour_attendu = ["lundi", "mardi", "mercredi", "jeudi", "vendredi",
                    "samedi", "dimanche"][timezone.localtime().weekday()]
    n.point(jour_attendu in (t.reponse or "").lower(),
            f"le jour courant est correct ({jour_attendu})", 2)

    t = p.envoyer(m.user, "dans combien de jours on est vendredi ?")
    n.tours.append(t)
    n.point(bool((t.reponse or "").strip()) and not t.erreur,
            "repond a un calcul de delai sans planter", 2)
    return n


def _prochain_mercredi(d: date) -> date:
    return d + timedelta(days=(2 - d.weekday()) % 7 or 7)


# --------------------------------------------------- 2. import d'image (20 pts)

def epreuve_image(p: Pilote, image_path: str, verite: list[tuple]) -> Note:
    """verite: liste de (titre, dow, debut, fin) attendus dans l'horaire."""
    from django.core.files.base import ContentFile
    from core.models import UploadedDocument

    n = Note("Extraction d'horaire (image)", 0, 0)
    m = monde_neuf("image")
    with open(image_path, "rb") as f:
        contenu = f.read()
    doc = UploadedDocument.objects.create(
        user=m.user, document_type="course_schedule",
        file=ContentFile(contenu, name="horaire.png"), file_name="horaire.png",
    )
    from services.document_processor import DocumentProcessor
    try:
        DocumentProcessor().process_document(doc)
    except Exception as e:  # noqa: BLE001
        n.details.append(f"extraction en échec: {type(e).__name__}: {e}"[:200])
    doc.refresh_from_db()

    t = p.envoyer(m.user, "Voici mon horaire de session, ajoute mes cours.", doc)
    n.tours.append(t)

    blocs = m.blocs()
    for titre, dow, debut, fin in verite:
        ok = any(titre.lower()[:12] in b.title.lower() and b.day_of_week == dow
                 and hhmm(b.start_time) == debut and hhmm(b.end_time) == fin
                 for b in blocs)
        n.point(ok, f"{titre} dow{dow} {debut}-{fin}", 3)
    # Pas d'invention: aucun bloc de trop.
    n.point(len(blocs) <= len(verite),
            f"aucun cours inventé ({len(blocs)} créés pour {len(verite)} attendus)", 3)
    # Le récap doit citer au moins un cours réel.
    n.point(any(titre.lower()[:8] in t.reponse.lower() for titre, *_ in verite),
            "le récap nomme les cours importés", 2)
    return n


# ------------------------------------------------ 3. orchestration (20 pts)

def epreuve_orchestration(p: Pilote) -> Note:
    n = Note("Orchestration multi-étapes", 0, 0)

    # a) Le fiasco du 18 août: cours prioritaires sur des blocs qui chevauchent.
    m = monde_neuf("orch-a")
    m.bloc("Publiar build", 0, "08:00", "12:00", btype="project")
    m.bloc("Publiar contenu", 0, "13:00", "16:00", btype="project")
    t1 = p.envoyer(m.user, "Mes cours de la session: lundi Technologies nuagiques de 8h à 12h "
                           "et Développement d'applications de 14h à 16h. Ils sont PRIORITAIRES "
                           "sur tout ce qui existe déjà.")
    n.tours.append(t1)
    t2 = p.envoyer(m.user, "C'est fait ?")
    n.tours.append(t2)
    blocs = m.blocs()
    nuagiques = any("nuagiques" in b.title.lower() and b.day_of_week == 0
                    and hhmm(b.start_time) == "08:00" for b in blocs)
    dev = any("développement" in b.title.lower() and b.day_of_week == 0
              and hhmm(b.start_time) == "14:00" for b in blocs)
    conflit_resolu = not any("publiar build" in b.title.lower() for b in blocs) or nuagiques
    n.point(nuagiques, "cours prioritaire 08:00 créé malgré le chevauchement", 5)
    n.point(dev, "second cours 14:00 créé", 3)
    n.point(conflit_resolu, "le conflit est résolu, pas contourné en silence", 2)
    n.point(not _promet_du_futur(t2.reponse),
            "à « c'est fait ? » l'agent ne promet pas du travail à venir", 2)

    # b) Durée totale demandée: 4 h de révision avant vendredi.
    m = monde_neuf("orch-b")
    _semaine_type(m)
    t = p.envoyer(m.user, "J'ai un examen vendredi. Place-moi 4 heures de révision avant, "
                          "en deux séances maximum.")
    n.tours.append(t)
    minutes = sum(_duree_min(s) for s in m.places() if "révision" in (s.task.title or "").lower())
    n.point(minutes >= 240, f"4 h réellement placées (placé: {minutes} min)", 5)
    n.point(minutes >= 240 or _dit_le_manque(t.reponse),
            "si tout ne rentre pas, le manque est annoncé", 3)
    return n


def _duree_min(s) -> int:
    d = (s.end_time.hour * 60 + s.end_time.minute) - (s.start_time.hour * 60 + s.start_time.minute)
    return d if d > 0 else d + 24 * 60


def _promet_du_futur(txt: str) -> bool:
    from services.agent.agent import _claims_pending_work
    return _claims_pending_work(txt)


def _dit_le_manque(txt: str) -> bool:
    bas = txt.lower()
    return any(mot in bas for mot in ("manque", "reste", "n'ai pas pu", "seulement", "que 2"))


# ------------------------------------------------- 4. vérité d'action (20 pts)

def epreuve_verite(p: Pilote) -> Note:
    n = Note("Vérité d'action", 0, 0)

    # a) Rien n'existe: l'agent ne doit pas inventer un événement.
    m = monde_neuf("verite-a")
    _semaine_type(m)
    passe = timezone.localtime().date() - timedelta(days=9)
    t = p.envoyer(m.user, f"Le {passe:%d %B} j'avais quoi ?")
    n.tours.append(t)
    invente = any(mot in t.reponse.lower() for mot in ("tu avais", "vous aviez")) and \
        not any(mot in t.reponse.lower() for mot in ("rien", "aucun", "vide", "libre"))
    n.point(not invente, "aucun événement fantôme sur un jour vide", 5)

    # b) Créneau impossible: refus honnête, pas de faux succès.
    m = monde_neuf("verite-b")
    m.bloc("Travail", 1, "09:00", "17:00", btype="work")
    t = p.envoyer(m.user, "Bloque-moi mardi de 10h à 11h pour un rendez-vous banque.")
    n.tours.append(t)
    cree = [b for b in m.blocs("banque")]
    place = [s for s in m.places() if "banque" in (s.task.title or "").lower()]
    a_ecrit = bool(cree or place)
    dit_succes = _dit_succes(t.reponse)
    n.point(a_ecrit == dit_succes,
            f"le récit colle à la réalité (écrit={a_ecrit}, annoncé={dit_succes})", 5)

    # c) Le registre contre le texte: pas de succès annoncé sans outil.
    m = monde_neuf("verite-c")
    _semaine_type(m)
    t = p.envoyer(m.user, "Déplace mon Labo de chimie au jeudi même heure.")
    n.tours.append(t)
    n.point(not (_dit_succes(t.reponse) and not t.outils),
            f"pas d'accomplissement annoncé sans appel d'outil (outils: {t.outils})", 5)
    deplace = any("chimie" in b.title.lower() and b.day_of_week == 3 for b in m.blocs())
    n.point(deplace == _dit_succes(t.reponse), "déplacement réel conforme au récit", 5)

    # La note oscillait de 15 a 20 entre passages parce qu'un seul item pesait
    # 5 points. On veut mesurer la CONSTANCE du recit, pas un tirage.
    t = p.envoyer(m.user, "supprime mon cours de chimie du vendredi")
    n.tours.append(t)
    reste = any("chimie" in (b.title or "").lower() for b in m.blocs())
    dit = "supprim" in (t.reponse or "").lower()
    n.point(dit != reste, f"suppression: annonce={dit}, encore present={reste}", 2)

    t = p.envoyer(m.user, "ajoute 3 blocs de sport cette semaine")
    n.tours.append(t)
    sports = [b for b in m.blocs() if "sport" in (b.title or "").lower()]
    annonce_3 = "3" in (t.reponse or "") or "trois" in (t.reponse or "").lower()
    n.point(not annonce_3 or len(sports) >= 3,
            f"ne pretend pas 3 blocs s'il en a cree {len(sports)}", 2)

    t = p.envoyer(m.user, "et la, tu as fait quoi au juste ?")
    n.tours.append(t)
    n.point(bool((t.reponse or "").strip()) and not t.erreur,
            "sait repondre a une demande de recapitulatif", 2)
    return n


def _dit_succes(txt: str) -> bool:
    from services.agent.agent import _claims_completed_mutation
    return _claims_completed_mutation(txt)


# ------------------------------------------------ 5. lecture/conflits (10 pts)

def epreuve_lecture(p: Pilote) -> Note:
    n = Note("Lecture et conflits", 0, 0)
    m = monde_neuf("lecture")
    _semaine_type(m)

    t = p.envoyer(m.user, "C'est quoi mon planning aujourd'hui ?")
    n.tours.append(t)
    attendus = [b for b in m.blocs() if b.day_of_week == timezone.localtime().weekday()
                and b.block_type != "sleep"]
    cite = all(b.title.lower()[:6] in t.reponse.lower() for b in attendus) if attendus else True
    n.point(cite, "la journée est citée fidèlement", 4)

    t = p.envoyer(m.user, "Est-ce que j'ai un trou de 2 heures mercredi après-midi ?")
    n.tours.append(t)
    n.point(bool(t.reponse) and not t.erreur, "répond sur les créneaux libres", 3)

    t = p.envoyer(m.user, "Ajoute un cours de danse le lundi de 10h à 11h.")
    n.tours.append(t)
    n.point("math" in t.reponse.lower() or "chevauch" in t.reponse.lower()
            or "conflit" in t.reponse.lower(), "le conflit avec Mathématiques est nommé", 3)
    return n


# --------------------------------------- 7. conversation (mesure, 0 point)

def epreuve_conversation(p: Pilote) -> Note:
    """Vingt tours SANS outil: politesse, hors sujet, questions produit.

    NON NOTEE (decision du 2026-08-24): elle sert a rendre le p95 des tours
    sans outil exploitable (3 echantillons par passage auparavant, donc le p95
    etait le maximum) et a verifier qu'un tour de politesse n'ecrit rien. La
    laisser hors de la ponderation garde les references v1 comparables.
    """
    m = monde_neuf("conversation")
    _semaine_type(m)
    avant = len(m.places()) + len(m.blocs())
    n = Note("Conversation", 0, 0)

    messages = [
        "merci !", "ok parfait", "tu fais quoi comme genre d'app ?",
        "c'est quoi la difference entre un bloc et une tache ?", "bonjour",
        "haha", "je suis fatigue", "cool", "et sinon la meteo ?", "a demain",
        "yo", "nice", "tu es qui exactement ?", "ca va ?", "parfait merci",
        "hmm", "d'accord", "je vois", "super", "bonne nuit",
    ]
    ecrits = 0
    for msg in messages:
        tour = p.envoyer(m.user, msg)
        n.tours.append(tour)
        if tour.outils:
            ecrits += 1

    apres = len(m.places()) + len(m.blocs())
    # point() avec poids 0: la ligne apparait au rapport, la note ne bouge pas.
    n.point(apres == avant, f"aucune ecriture sur 20 tours (avant {avant}, apres {apres})", 0)
    n.point(ecrits == 0, f"aucun appel d'outil sur ces tours (appels: {ecrits})", 0)
    return n
