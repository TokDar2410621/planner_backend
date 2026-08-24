"""
Génère une image d'horaire de session dont on connaît la VÉRITÉ TERRAIN.

Une capture réelle ferait un meilleur test de vision, mais sa vérité serait
discutable: ici chaque case est dessinée depuis la table VERITE, donc la
notation est indiscutable et rejouable à l'identique pour v1 et v2.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# (titre, dow lundi=0, début, fin, salle)
VERITE = [
    ("Technologies nuagiques", 0, "08:00", "12:00", "316.1"),
    ("Développement d'applications", 0, "14:00", "16:00", "314.1"),
    ("Conception d'applications", 1, "08:00", "11:00", "314.1"),
    ("Réseaux industriels", 2, "13:00", "16:00", "337.1"),
    ("Téléphonie I", 3, "09:00", "12:00", "612.2"),
]

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
H_DEBUT, H_FIN = 8, 18
LARG_COL, HAUT_H, MARGE_G, MARGE_H = 220, 60, 90, 70


def _police(taille: int):
    for chemin in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Users/Darius/.claude/skills/aso-appstore-screenshots/assets/ArchivoBlack-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(chemin, taille)
        except OSError:
            continue
    return ImageFont.load_default()


def generer(chemin: str) -> str:
    larg = MARGE_G + LARG_COL * len(JOURS) + 20
    haut = MARGE_H + HAUT_H * (H_FIN - H_DEBUT) + 20
    img = Image.new("RGB", (larg, haut), "white")
    d = ImageDraw.Draw(img)
    f_titre, f_case, f_petit = _police(22), _police(17), _police(14)

    for i, jour in enumerate(JOURS):
        d.text((MARGE_G + i * LARG_COL + 12, 24), jour, fill="black", font=f_titre)
    for h in range(H_DEBUT, H_FIN + 1):
        y = MARGE_H + (h - H_DEBUT) * HAUT_H
        d.line([(MARGE_G, y), (larg - 20, y)], fill="#cccccc")
        d.text((16, y - 10), f"{h:02d}:00", fill="#444444", font=f_petit)
    for i in range(len(JOURS) + 1):
        x = MARGE_G + i * LARG_COL
        d.line([(x, MARGE_H), (x, haut - 20)], fill="#cccccc")

    for titre, dow, debut, fin, salle in VERITE:
        hd, md = map(int, debut.split(":"))
        hf, mf = map(int, fin.split(":"))
        y1 = MARGE_H + (hd - H_DEBUT + md / 60) * HAUT_H
        y2 = MARGE_H + (hf - H_DEBUT + mf / 60) * HAUT_H
        x1 = MARGE_G + dow * LARG_COL + 4
        x2 = x1 + LARG_COL - 8
        d.rounded_rectangle([x1, y1 + 3, x2, y2 - 3], radius=8,
                            fill="#dbeafe", outline="#3b82f6", width=2)
        for j, ligne in enumerate(_couper(titre, 18)):
            d.text((x1 + 10, y1 + 12 + j * 22), ligne, fill="#111111", font=f_case)
        d.text((x1 + 10, y2 - 30), f"{debut}-{fin} · {salle}", fill="#334155", font=f_petit)

    img.save(chemin, "PNG")
    return chemin


def _couper(texte: str, n: int) -> list[str]:
    mots, lignes, courante = texte.split(), [], ""
    for mot in mots:
        essai = f"{courante} {mot}".strip()
        if len(essai) <= n:
            courante = essai
        else:
            lignes.append(courante)
            courante = mot
    if courante:
        lignes.append(courante)
    return lignes[:3]


def verite_pour_notation() -> list[tuple]:
    return [(t, d, deb, fin) for t, d, deb, fin, _ in VERITE]
