"""
Lanceur du banc: `python -m benchmarks.run v1` (ou v2, ou les deux).

Fait de VRAIS appels LLM. Écrit dans la base configurée: à lancer avec une
base JETABLE (sans DATABASE_URL, Django tombe sur SQLite local).
Rend un score sur 100 par agent, plus le détail épreuve par épreuve, et
enregistre un rapport horodaté dans docs/.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import django

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "planner.settings")
django.setup()

from benchmarks import epreuves, juge  # noqa: E402
from benchmarks.harness import Note, Pilote, pilote_v1, pilote_v2  # noqa: E402
from benchmarks.horaire_image import generer, verite_pour_notation  # noqa: E402

# La somme des poids EST le maximum de score_sur_100. Une epreuve absente de ce
# dict pese zero: c'est ainsi que "Conversation" mesure sans noter, et que les
# references v1 (78,3 / 85,7 / 90,0) restent comparables.
PONDERATION = {
    "Compréhension temporelle": 20,
    "Extraction d'horaire (image)": 20,
    "Orchestration multi-étapes": 20,
    "Vérité d'action": 20,
    "Lecture et conflits": 10,
    "Ton et format": 10,
}


def passer_examen(p: Pilote, image: str) -> list[Note]:
    notes = []
    for libelle, fn in (
        ("temporel", lambda: epreuves.epreuve_temporel(p)),
        ("image", lambda: epreuves.epreuve_image(p, image, verite_pour_notation())),
        ("orchestration", lambda: epreuves.epreuve_orchestration(p)),
        ("verite", lambda: epreuves.epreuve_verite(p)),
        ("lecture", lambda: epreuves.epreuve_lecture(p)),
        ("ton", lambda: juge.epreuve_ton(p)),
        ("conversation", lambda: epreuves.epreuve_conversation(p)),
    ):
        print(f"  [{p.nom}] épreuve {libelle}…", flush=True)
        try:
            notes.append(fn())
        except Exception as e:  # noqa: BLE001
            n = Note(_nom_epreuve(libelle), 0, 1)
            n.details.append(f"épreuve en échec: {type(e).__name__}: {e}"[:300])
            notes.append(n)
    return notes


def _nom_epreuve(cle: str) -> str:
    return {
        "temporel": "Compréhension temporelle",
        "image": "Extraction d'horaire (image)",
        "orchestration": "Orchestration multi-étapes",
        "verite": "Vérité d'action",
        "lecture": "Lecture et conflits",
        "ton": "Ton et format",
    }[cle]


def score_sur_100(notes: list[Note]) -> float:
    total = 0.0
    for n in notes:
        poids = PONDERATION.get(n.epreuve, 0)
        total += poids * (n.pourcent / 100.0)
    return round(total, 1)


def rapport(nom: str, notes: list[Note], lat: dict | None = None) -> str:
    lignes = [f"## Agent {nom}: {score_sur_100(notes)} / 100", ""]
    if lat:
        a, sa = lat["avec_outils"], lat["sans_outils"]
        lignes += [
            "### Latence (secondes par tour)",
            f"- avec outils (n={a['n']}): p50 {a['p50']}, **p95 {a['p95']}**",
            f"- sans outils (n={sa['n']}): p50 {sa['p50']}, **p95 {sa['p95']}**",
            "",
        ]
    for n in notes:
        poids = PONDERATION.get(n.epreuve, 0)
        obtenu = round(poids * n.pourcent / 100.0, 1)
        lignes.append(f"### {n.epreuve}: {obtenu} / {poids}")
        lignes += [f"- {d}" for d in n.details]
        for t in n.tours:
            if t.erreur:
                lignes.append(f"  - ERREUR sur « {t.message[:50]} »: {t.erreur}")
        lignes.append("")
    return "\n".join(lignes)


def main() -> None:
    cibles = [a for a in sys.argv[1:] if a in ("v1", "v2")] or ["v1"]
    image = generer(os.path.join(os.path.dirname(__file__), "horaire_test.png"))
    print(f"Image d'épreuve: {image}")

    # --fournisseur=deepseek force les DEUX agents sur le meme modele, ce qui
    # isole l'apport de la boucle. Sans lui, on mesure le produit tel quel.
    force = next((a.split("=", 1)[1] for a in sys.argv[1:]
                  if a.startswith("--fournisseur=")), None)
    if force:
        from benchmarks import harness
        harness.FOURNISSEUR = force
        print(f"Fournisseur force: {force}")

    blocs = []
    for cible in cibles:
        p = pilote_v1() if cible == "v1" else pilote_v2()
        print(f"\n=== Examen de l'agent {cible} ===", flush=True)
        notes = passer_examen(p, image)
        blocs.append(rapport(cible, notes, p.latences()))
        print(f"  -> {score_sur_100(notes)} / 100")

    horodatage = datetime.now().strftime("%Y-%m-%d-%H%M")
    chemin = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "docs", f"banc-agent-{horodatage}.md",
    )
    entete = (f"# Banc d'évaluation de l'agent Planner\n\n"
              f"Passé le {datetime.now():%Y-%m-%d à %H:%M}. Vrais appels LLM, base jetable.\n\n")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(entete + "\n".join(blocs))
    print(f"\nRapport: {chemin}")


if __name__ == "__main__":
    main()
