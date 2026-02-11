"""
Interactive UI tools - Let the AI present structured forms and choices to the user.
"""
from django.contrib.auth.models import User
from .base import BaseTool, ToolResult


class PresentFormTool(BaseTool):
    """
    Present an interactive form to the user with various input types.

    The frontend renders these as rich UI components (time pickers, checkboxes, etc.)
    The user's selections are sent back as the next message.
    """

    name = "present_form"
    description = (
        "Présente un formulaire interactif à l'utilisateur avec des champs structurés. "
        "Utilise cet outil quand tu as besoin que l'utilisateur fasse des choix précis "
        "(horaires, jours de la semaine, options multiples, etc.) au lieu de taper du texte libre. "
        "Les types disponibles : time_range (plage horaire), time (heure), number (nombre), "
        "checkbox (choix multiples), select (liste déroulante), radio (choix unique avec option 'Autre')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "description": "Liste des champs du formulaire",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Identifiant unique du champ (ex: 'sleep_time', 'work_days')"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["time_range", "time", "number", "checkbox", "select", "radio"],
                            "description": "Type de champ: time_range (plage horaire start/end), time (heure unique), number (nombre), checkbox (choix multiples), select (dropdown), radio (choix unique)"
                        },
                        "label": {
                            "type": "string",
                            "description": "Label court du champ (ex: 'Heures de sommeil')"
                        },
                        "question": {
                            "type": "string",
                            "description": "Question affichée au-dessus du champ (ex: 'À quelle heure tu te couches et te réveilles ?')"
                        },
                        "options": {
                            "type": "array",
                            "description": "Options pour checkbox/select/radio. Chaque option a un value et un label.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "label": {"type": "string"}
                                },
                                "required": ["value", "label"]
                            }
                        },
                        "default": {
                            "description": "Valeur par défaut. Pour time_range: {start: '23:00', end: '07:00'}. Pour time: '18:00'. Pour number: 8."
                        },
                        "min": {
                            "type": "number",
                            "description": "Valeur minimum (pour type number)"
                        },
                        "max": {
                            "type": "number",
                            "description": "Valeur maximum (pour type number)"
                        },
                        "allow_other": {
                            "type": "boolean",
                            "description": "Pour radio: afficher une option 'Autre' avec texte libre (défaut: true)"
                        },
                        "other_placeholder": {
                            "type": "string",
                            "description": "Placeholder pour l'option 'Autre' du radio"
                        }
                    },
                    "required": ["id", "type", "label", "question"]
                }
            }
        },
        "required": ["inputs"]
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        inputs = kwargs.get("inputs", [])

        if not inputs:
            return ToolResult(
                success=False,
                message="Aucun champ spécifié pour le formulaire."
            )

        # Validate and normalize inputs
        normalized = []
        for inp in inputs:
            field = {
                "id": inp["id"],
                "type": inp["type"],
                "label": inp["label"],
                "question": inp["question"],
            }

            # Add options for choice types
            if inp["type"] in ("checkbox", "select", "radio"):
                field["options"] = inp.get("options", [])
                if not field["options"]:
                    return ToolResult(
                        success=False,
                        message=f"Le champ '{inp['id']}' de type {inp['type']} nécessite des options."
                    )

            # Add defaults
            if "default" in inp:
                field["default"] = inp["default"]

            # Number constraints
            if inp["type"] == "number":
                if "min" in inp:
                    field["min"] = inp["min"]
                if "max" in inp:
                    field["max"] = inp["max"]

            # Radio specific
            if inp["type"] == "radio":
                if "allow_other" in inp:
                    field["allowOther"] = inp["allow_other"]
                if "other_placeholder" in inp:
                    field["otherPlaceholder"] = inp["other_placeholder"]

            normalized.append(field)

        return ToolResult(
            success=True,
            data={"interactive_inputs": normalized},
            message=f"Formulaire avec {len(normalized)} champ(s) présenté à l'utilisateur. Attends sa réponse."
        )


class PresentQuickRepliesTool(BaseTool):
    """
    Present quick reply buttons to the user.
    """

    name = "present_quick_replies"
    description = (
        "Présente des boutons de réponse rapide à l'utilisateur. "
        "Utilise cet outil quand tu veux proposer 2-4 actions rapides après ta réponse. "
        "Chaque bouton a un label (texte affiché) et une value (message envoyé au clic)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "replies": {
                "type": "array",
                "description": "Liste des boutons (max 4)",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Texte affiché sur le bouton (court, avec emoji optionnel)"
                        },
                        "value": {
                            "type": "string",
                            "description": "Message envoyé quand l'utilisateur clique"
                        }
                    },
                    "required": ["label", "value"]
                }
            }
        },
        "required": ["replies"]
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        replies = kwargs.get("replies", [])

        if not replies:
            return ToolResult(
                success=False,
                message="Aucun bouton spécifié."
            )

        # Limit to 4 buttons
        normalized = replies[:4]

        return ToolResult(
            success=True,
            data={"quick_replies": normalized},
            message=f"{len(normalized)} bouton(s) de réponse rapide présenté(s)."
        )
