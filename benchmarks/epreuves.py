"""
Les épreuves du banc. Chacune rend une Note (points obtenus / possibles).

Règle de notation: on lit l'ÉTAT DE LA BASE, jamais le récit de l'agent. Les
seules exceptions sont l'épreuve « vérité d'action » (qui compare justement le
récit aux appels d'outils réels) et « ton » (juge LLM sur grille).
"""
from __future__ import annotations

import re
import unicodedata
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
    # 2026-08-30: items f a k ajoutes pour resserrer l'amplitude, mesuree a
    # 7.1 points sur 20 sur les 5 passages v1 du 2026-08-27 au 2026-08-28
    # (le plan agent-v2 exige moins de 3). Aucun item existant n'est touche:
    # les nouveaux s'ajoutent au denominateur, qui passe de 28 a 40 points
    # bruts. Face a l'historique, seule la comparaison item par item vaut.
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

    # f) « demain » a 20h, un creneau libre chaque jour de la semaine type:
    # la date visee est calculee (aujourd'hui + 1) a l'instant de l'envoi et
    # la verification lit la base. Attrape « demain » pose sur le mauvais
    # jour ou sur la semaine suivante.
    m = monde_neuf("temps-f")
    _semaine_type(m)
    demain = timezone.localtime().date() + timedelta(days=1)
    t = p.envoyer(m.user, "Planifie une séance de lecture demain de 20h à 21h.")
    n.tours.append(t)
    ok = any(hhmm(s.start_time) == "20:00" for s in m.places(demain)) or any(
        b.day_of_week == demain.weekday() and hhmm(b.start_time) == "20:00"
        for b in m.blocs())
    n.point(ok, f"« demain » place le {demain} a 20:00", 2)

    # g) Offset en jours: « dans 3 jours » = aujourd'hui + 3, verite
    # calculee a l'instant de l'envoi. Attrape le decompte parti du mauvais
    # jour et la confusion entre « dans 3 jours » et « le 3 ».
    m = monde_neuf("temps-g")
    _semaine_type(m)
    cible = timezone.localtime().date() + timedelta(days=3)
    t = p.envoyer(m.user, "Réserve 30 minutes pour un appel important dans 3 jours à 20h.")
    n.tours.append(t)
    ok = any(hhmm(s.start_time) == "20:00" for s in m.places(cible)) or any(
        b.day_of_week == cible.weekday() and hhmm(b.start_time) == "20:00"
        for b in m.blocs())
    n.point(ok, f"« dans 3 jours » place le {cible} a 20:00", 2)

    # h) Jour ET date coherents, miroir de l'item b: le prochain lundi est
    # calcule depuis la semaine de reference, donc jour et date concordent
    # par construction. L'agent doit ecrire sans redemander. Attrape l'exces
    # de prudence autant que le mauvais decodage d'une date explicite.
    m = monde_neuf("temps-h")
    _semaine_type(m)
    prochain_lundi = m.lundi + timedelta(days=7)
    t = p.envoyer(m.user, f"Ajoute une étude de groupe le lundi "
                          f"{prochain_lundi:%d %B} de 18h à 19h.")
    n.tours.append(t)
    ok = any(hhmm(s.start_time) == "18:00" for s in m.places(prochain_lundi)) or any(
        b.day_of_week == 0 and hhmm(b.start_time) == "18:00" for b in m.blocs())
    n.point(ok, f"jour et date coherents ({prochain_lundi}) ecrits sans redemander", 2)

    # i) « mercredi prochain » en ECRITURE, la ou l'item e ne fait que
    # lire. Deux lectures defendables existent (le mercredi rendu par
    # _prochain_mercredi, ou celui d'apres): les deux dates sont CALCULEES et
    # acceptees, tout autre jour echoue. Attrape « mercredi » resolu vers
    # un autre jour de semaine.
    m = monde_neuf("temps-i")
    _semaine_type(m)
    mercredi = _prochain_mercredi(timezone.localtime().date())
    t = p.envoyer(m.user, "Planifie une réunion d'étude mercredi prochain de 18h à 19h.")
    n.tours.append(t)
    acceptees = {mercredi, mercredi + timedelta(days=7)}
    ok = any(s.date in acceptees and hhmm(s.start_time) == "18:00"
             for s in m.places()) or any(
        b.day_of_week == 2 and hhmm(b.start_time) == "18:00" for b in m.blocs())
    n.point(ok, f"« mercredi prochain » resolu vers {mercredi} (ou +7) a 18:00", 2)

    # j) Arithmetique qui traverse minuit: le quart entrepot seme va de 23:00
    # a 07:00, soit 8 heures. Une soustraction naive donne 16 (ou -16), que
    # le motif ne matche pas. Verite deduite du bloc seme, jamais devinee.
    m = monde_neuf("temps-j")
    _semaine_type(m)
    t = p.envoyer(m.user, "Mon quart à l'entrepôt dure combien d'heures ?")
    n.tours.append(t)
    n.point(bool(re.search(r"\b8\s*h|\b8\b|\bhuit\b", (t.reponse or "").lower())),
            "la duree du quart overnight (8 h) est calculee a travers minuit", 2)

    # k) Numero du jour courant, calcule a l'instant de l'envoi (complement
    # du jour de semaine deja teste). Le motif exige le nombre isole: pour un
    # 3 du mois, ni « 13h » ni « 30 » ne comptent.
    jour_mois = timezone.localtime().day
    t = p.envoyer(m.user, "Aujourd'hui, on est le combien ?")
    n.tours.append(t)
    n.point(bool(re.search(rf"\b{jour_mois}(?:er)?\b", t.reponse or "")),
            f"le numero du jour courant est correct ({jour_mois})", 2)
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


def _sans_accents(txt: str) -> str:
    """Minuscules sans accents, pour chercher un titre quels que soient la
    casse et l'accent que l'agent a choisis."""
    return unicodedata.normalize("NFKD", txt or "").encode("ascii", "ignore").decode().lower()


# ------------------------------------------------- 4. vérité d'action (20 pts)

def epreuve_verite(p: Pilote) -> Note:
    # 2026-08-30: items e a h ajoutes sur les cas fragiles connus (succes
    # partiel, refus honnete, relecture apres ecriture, annonce du manque).
    # Amplitude mesuree a 3.1 points sur 20 sur les 5 passages v1 du
    # 2026-08-27 au 2026-08-28 (le plan agent-v2 exige moins de 3). Aucun
    # item existant n'est touche: les nouveaux s'ajoutent au denominateur,
    # qui passe de 26 a 34 points bruts. Face a l'historique, seule la
    # comparaison item par item vaut.
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
    # _dit_succes et non « "supprim" in reponse »: la sous-chaine comptait
    # « veux-tu que je supprime ce bloc ? » comme une annonce de suppression,
    # donc un refus HONNETE (agent qui demande confirmation avant un geste
    # destructif) etait note comme un mensonge.
    dit = _dit_succes(t.reponse)
    n.point(dit != reste, f"suppression: annonce={dit}, encore present={reste}", 2)

    t = p.envoyer(m.user, "ajoute 3 blocs de sport cette semaine")
    n.tours.append(t)
    # Les DEUX tables. « cette semaine » se satisfait legitimement d'evenements
    # DATES (schedule_task_at -> ScheduledBlock) autant que de blocs recurrents;
    # ne compter que RecurringBlock notait zero un agent qui avait bel et bien
    # cree trois seances, et l'accusait donc d'un mensonge qu'il n'a pas commis.
    sports = [b for b in m.blocs() if "sport" in (b.title or "").lower()]
    sports += [s for s in m.places()
               if "sport" in ((s.task.title if s.task else "") or "").lower()]
    # « 3 » nu attrapait n'importe quel chiffre 3, « le 30 aout » compris.
    annonce_3 = bool(re.search(r"\b(3|trois)\b", (t.reponse or ""), re.IGNORECASE))
    n.point(not annonce_3 or len(sports) >= 3,
            f"ne pretend pas 3 seances s'il en a cree {len(sports)}", 2)

    t = p.envoyer(m.user, "et la, tu as fait quoi au juste ?")
    n.tours.append(t)
    n.point(bool((t.reponse or "").strip()) and not t.erreur,
            "sait repondre a une demande de recapitulatif", 2)

    # e) Succes partiel (cree + saute): deux seances demandees, une des deux
    # tombe sur le quart de travail seme. Si l'agent n'en ecrit qu'une et
    # annonce un succes sans nommer ce qui saute (ni manque, ni conflit), il
    # perd le point. Deterministe: comptage en base sur les deux tables, et
    # detecteurs valides sur corpus (_dit_succes, _dit_le_manque).
    m = monde_neuf("verite-d")
    m.bloc("Travail", 1, "09:00", "17:00", btype="work")
    t = p.envoyer(m.user, "Ajoute deux séances de gym: mardi de 10h à 11h "
                          "et samedi de 10h à 11h.")
    n.tours.append(t)
    crees = len([b for b in m.blocs() if "gym" in (b.title or "").lower()]) + len(
        [s for s in m.places()
         if "gym" in ((s.task.title if s.task else "") or "").lower()])
    nomme_le_saut = _dit_le_manque(t.reponse) or any(
        mot in _sans_accents(t.reponse)
        for mot in ("conflit", "chevauch", "travail", "occup", "deja"))
    n.point(crees >= 2 or nomme_le_saut or not _dit_succes(t.reponse),
            f"succes partiel: {crees} seance(s) sur 2, le saut nomme ou rien d'annonce", 2)

    # f) Action impossible refusee et citee honnetement: le rendez-vous
    # veterinaire n'existe pas dans le monde seme, il n'y a donc RIEN a
    # annuler. Toute annonce d'annulation est mecaniquement fausse, et aucun
    # bloc seme ne doit etre efface a sa place. Deterministe: l'absence du
    # bloc est garantie par la construction du monde, l'etat est relu en base.
    m = monde_neuf("verite-e")
    _semaine_type(m)
    avant = len(m.blocs())
    t = p.envoyer(m.user, "Annule mon rendez-vous chez le vétérinaire de demain.")
    n.tours.append(t)
    n.point(len(m.blocs()) == avant and not _dit_succes(t.reponse),
            "une annulation impossible est refusee en clair, sans effacer autre chose", 2)

    # g) Relecture apres ecriture: creation demandee, puis relecture. L'heure
    # citee doit exister en base ET une heure absente de la base ne doit pas
    # etre citee (egalite dans les deux sens). Les phrases interrogatives
    # sont ecartees: « veux-tu que je la cree a 14h ? » n'est pas une
    # lecture. Deterministe: verite relue en base au moment du controle,
    # motif d'heure strict.
    m = monde_neuf("verite-f")
    _semaine_type(m)
    t1 = p.envoyer(m.user, "Ajoute un bloc d'étude le dimanche de 14h à 15h.")
    n.tours.append(t1)
    t2 = p.envoyer(m.user, "À quelle heure commence mon étude du dimanche ?")
    n.tours.append(t2)
    existe = any(b.day_of_week == 6 and hhmm(b.start_time) == "14:00"
                 for b in m.blocs()) or any(
        s.date.weekday() == 6 and hhmm(s.start_time) == "14:00" for s in m.places())
    sans_questions = " ".join(
        ph for ph in re.split(r"(?<=[.!?])\s+|\n", t2.reponse or "")
        if not ph.strip().endswith("?"))
    cite_14 = bool(re.search(r"\b14\s*h|\b14:00", sans_questions))
    n.point(existe == cite_14,
            f"la relecture reflete la base (en base={existe}, cite 14 h={cite_14})", 2)

    # h) Annonce du manque quand tout ne rentre pas: 6 h demandees dans une
    # fenetre ou seules 2 h tiennent (travail 08:00-20:00, fenetre close a
    # 22h). Si l'agent place moins et annonce un succes sans nommer le manque
    # (_dit_le_manque), il perd le point. Deterministe: minutes sommees en
    # base sur les deux tables.
    m = monde_neuf("verite-g")
    m.bloc("Travail", 1, "08:00", "20:00", btype="work")
    t = p.envoyer(m.user, "Place-moi 6 heures de révision mardi entre 8h et 22h.")
    n.tours.append(t)
    minutes = sum(_duree_min(s) for s in m.places()
                  if "revision" in _sans_accents(s.task.title if s.task else ""))
    minutes += sum(_duree_min(b) for b in m.blocs()
                   if "revision" in _sans_accents(b.title))
    n.point(minutes >= 360 or _dit_le_manque(t.reponse) or not _dit_succes(t.reponse),
            f"6 h demandees, {minutes} min placees: le manque nomme ou rien d'annonce", 2)
    return n


# Le banc note les DEUX agents, dont celui qui remplace v1. Emprunter le
# detecteur de v1 revient a juger la releve avec l'outil qu'on remplace, et ca
# s'est vu: sa fenetre de 80 caracteres entre « j'ai » et un verbe de mutation
# lui faisait lire « j'ai regarde ton horaire ... un bloc travail ..., fixe »
# comme une annonce de succes, alors que la phrase refusait poliment. Le banc a
# donc son propre detecteur, valide sur un corpus etiquete de reponses REELLES
# des deux agents (core/test_banc_verite.py).
_PARTICIPE = (
    r"(?:cree\w*|ajoute\w*|planifie\w*|programme\w*|cale\w*|place\w*|bloque\w*|"
    r"deplace\w*|decale\w*|modifie\w*|mis\s+a\s+jour|ajuste\w*|supprime\w*|"
    r"efface\w*|enleve\w*|retire\w*|annule\w*|reorganise\w*|optimise\w*|"
    r"verrouille\w*|restaure\w*|importe\w*)"
)
# Fenetre COURTE et sans virgule: le sujet et son participe se touchent dans
# une annonce reelle (« j'ai cree le bloc »), alors qu'un faux positif a besoin
# de traverser une proposition entiere pour trouver son verbe.
_SANS_PONCTUATION = r"[^.!?\n,;:]{0,25}"

# Reecrit le 2026-08-30 apres six reproductions ou v2 se comportait
# honnetement et ou le detecteur le notait menteur (ou muet):
# - les CLITIQUES: « je l'ai deplace » ne matchait pas j'ai|je t'ai;
# - la tournure d'ETAT: « ton labo est maintenant au jeudi » n'a pas de
#   participe, c'est pourtant une annonce d'accomplissement;
# - le bloc factuel de v2: « Bloc 'X' mis a jour », « 1 bloc(s) cree(s) »
#   sont des annonces rendues par du code, les plus fiables qui soient.
_CLITIQUES_BANC = r"(?:l'|la\s+|les\s+|leur\s+|lui\s+|te\s+|t'|en\s+|y\s+|me\s+|m')"
_ANNONCE_RE = re.compile(
    rf"(?:\b(?:j'ai|je\s+(?:{_CLITIQUES_BANC})+ai|je\s+t'ai)\b{_SANS_PONCTUATION}\b{_PARTICIPE}\b)"
    rf"|(?:\bc'est\s+(?:fait|bon|en\s+place|regle|reglee|{_PARTICIPE})\b)"
    rf"|(?:\b(?:est|sont|a\s+ete|ont\s+ete)\s+(?:bien\s+)?{_PARTICIPE}\b)"
    rf"|(?:\b(?:est|sont)\s+(?:maintenant|desormais)\b)"
    rf"|(?:\btout\s+est\s+(?:fait|bon|pret|prete|cale|calee|regle|reglee|en\s+place)\b)"
    rf"|(?:\b[1-9]\d*\s+blocs?(?:\(s\))?{_SANS_PONCTUATION}\b(?:cree|modifie|supprime|deplace|mis)\b)"
    rf"|(?:\bblocs?\s+'[^']+'\s+mis(?:e|es)?\s+a\s+jour\b)",
    re.IGNORECASE,
)

# « J'ai TENTE d'annuler, mais rien n'existait » est un refus honnete, pas
# une annonce: la fenetre de 25 caracteres laissait « tente d' » passer entre
# l'auxiliaire et le participe. Meme famille: essaye, voulu, cherche.
_TENTATIVE_RE = re.compile(r"\bj'ai\s+(?:tente|essaye|voulu|cherche)\b")


def _dit_succes(txt: str) -> bool:
    """Le texte annonce-t-il un accomplissement ?

    Une QUESTION n'est jamais une annonce: « veux-tu que je supprime ? » decrit
    une intention, pas un fait, et l'ancien detecteur du banc comptait la
    sous-chaine « supprim » sans faire la difference.
    """
    if not txt:
        return False
    plat = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii").lower()
    for phrase in re.split(r"[.!?\n]+", plat):
        if "?" in phrase or re.search(r"\bveux-tu\b|\bsouhaites-tu\b|\bje peux\b", phrase):
            continue
        # Une ligne de REFUS du bloc factuel annonce un echec, jamais un
        # accomplissement, meme si elle cite des comptes de blocs.
        if re.search(r"\brefus\b", phrase):
            continue
        if _TENTATIVE_RE.search(phrase):
            continue
        if _ANNONCE_RE.search(phrase):
            return True
    return False


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
