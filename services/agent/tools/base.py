"""Types partages des outils fonctions.

Les outils sont des modules (NAME, DESCRIPTION, PARAMETERS, execute),
style OpenCode. Ce module ne porte que le contrat de resultat et les
issues metier. Il n'y a plus de classe de base : le harnais
(harnais.py) applique la validation et le contrat d'erreur autour de
chaque appel.
"""
from dataclasses import dataclass, field


# Issues metier partagees (audit 2026-10-13). Le booleen `success` ne decrit
# pas l'issue metier: une confirmation en attente, une cible absente et une
# panne valent toutes success=False. Reference: OpenCode Tool.Result, qui
# porte a la fois un `output` structure et un `content` lisible ecrit par
# l'outil lui-meme.
ISSUE_TERMINEE = "terminee"
ISSUE_PARTIELLE = "partielle"
ISSUE_AUCUN_CHANGEMENT = "aucun_changement"
ISSUE_BESOIN_PRECISION = "besoin_precision"
ISSUE_BESOIN_CONFIRMATION = "besoin_confirmation"
ISSUE_CAPACITE_ABSENTE = "capacite_absente"
ISSUE_ECHEC = "echec"
ISSUE_EFFET_A_VERIFIER = "effet_a_verifier"

ISSUES_METIER = frozenset({
    ISSUE_TERMINEE, ISSUE_PARTIELLE, ISSUE_AUCUN_CHANGEMENT,
    ISSUE_BESOIN_PRECISION, ISSUE_BESOIN_CONFIRMATION,
    ISSUE_CAPACITE_ABSENTE, ISSUE_ECHEC, ISSUE_EFFET_A_VERIFIER,
})


@dataclass
class ToolResult:
    """Resultat d'execution d'un outil.

    `success`/`data`/`message` restent pour compatibilite. `issue` dit
    l'issue metier (voir ISSUES_METIER), `cause` un code de cause stable,
    `public` l'explication affichable ecrite PAR L'OUTIL: le rendu la relaie
    telle quelle, il ne l'invente pas et ne recopie jamais `message` brut
    (qui peut contenir une exception ou une instruction).
    """

    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""
    issue: str = ""
    cause: str = ""
    public: str = ""

    def to_dict(self) -> dict:
        """Serialize for sending back to the LLM."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "issue": self.issue,
            "cause": self.cause,
            "public": self.public,
        }

    def to_string(self) -> str:
        """Convert to a string for the LLM to read."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
