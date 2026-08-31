"""
Send Web Push reminders + departure alerts for recurring blocks and tasks.

Boucle worker (railway.toml): toutes les ~10 min avec --lead 15. La fenêtre
regarde EN ARRIÈRE (12 min) et EN AVANT (--lead): les fenêtres consécutives se
chevauchent volontairement, et le journal PushSendLog garantit UN push par
occurrence. Sans ça, un bloc démarrant à l'heure pile pouvait tomber dans la
fissure entre deux ticks et n'être JAMAIS notifié (vécu 2026-08-09: une seule
notification sur toute une journée de blocs).

Trois passes:
  1. "Bientôt : <bloc>" quand un bloc récurrent démarre dans la fenêtre.
  2. Alertes de départ des blocs liés à un lieu (préparation + départ).
  3. Alertes de départ des tâches planifiées liées à un lieu.

Degrades silently if push is not configured.
"""
from datetime import datetime, time as dt_time, timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import PushSendLog, PushSubscription, RecurringBlock, ScheduledBlock
from services.commute import commute_window
from services.apns import rattraper_tous_les_reveils
from services.push import push_configured, send_to_user

# Rétro-regard: assez large pour couvrir la dérive des ticks (durée d'exécution
# qui s'ajoute au sleep), plus courte que l'intervalle de deux ticks + lead.
LOOKBACK_MINUTES = 12


def _fmt(minutes: int) -> str:
    """Minutes-since-midnight -> 'HH:MM' (clamped to the day)."""
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _window_segments(now, lead: int):
    """Découpe [now - LOOKBACK, now + lead] en segments par jour calendaire.

    Chaque segment: (date_occurrence, heure_debut, heure_fin). Deux segments
    quand la fenêtre traverse minuit (l'ancien code SAUTAIT ces ticks: les
    blocs de début de nuit n'étaient jamais rappelés).
    """
    start_dt = now - timedelta(minutes=LOOKBACK_MINUTES)
    end_dt = now + timedelta(minutes=lead)
    segments = []
    day = start_dt.date()
    while day <= end_dt.date():
        seg_start = start_dt.timetz() if day == start_dt.date() else dt_time.min
        seg_end = end_dt.timetz() if day == end_dt.date() else dt_time.max
        segments.append((day, seg_start, seg_end))
        day += timedelta(days=1)
    return segments


def _once(user, kind: str, ref, occ_date) -> bool:
    """True si ce push n'a pas encore été envoyé pour cette occurrence."""
    _, created = PushSendLog.objects.get_or_create(
        user=user, kind=kind, ref=str(ref), date=occ_date
    )
    return created


class Command(BaseCommand):
    help = "Push reminders + departure alerts for blocks within --lead minutes."

    def add_arguments(self, parser):
        parser.add_argument("--lead", type=int, default=15, help="Minutes ahead to remind.")

    def handle(self, *args, **opts):
        # Filet de securite du reveil silencieux iOS: les reveils demandes
        # qu'aucun Timer n'a rattrapes (processus web redemarre entre-temps).
        try:
            rattraper_tous_les_reveils()
        except Exception as e:  # noqa: BLE001
            self.stderr.write(f"Rattrapage des reveils APNs: {e}")

        if not push_configured():
            self.stdout.write("Web push not configured; nothing sent.")
            return

        lead = opts["lead"]
        now = timezone.localtime()
        now_min = now.hour * 60 + now.minute
        win_start_min = max(0, now_min - LOOKBACK_MINUTES)
        end_min = min(24 * 60 - 1, now_min + lead)

        from services.webhooks import dispatch

        user_ids = list(
            PushSubscription.objects.values_list("user_id", flat=True).distinct()
        )

        # --- Pass 1: "Bientôt" reminders (block starting within the window) ----
        matched = 0
        sent = 0
        for occ_date, seg_start, seg_end in _window_segments(now, lead):
            blocks = RecurringBlock.objects.filter(
                RecurringBlock.bounds_filter(occ_date),
                user_id__in=user_ids,
                active=True,
                day_of_week=occ_date.weekday(),
                start_time__gte=seg_start,
                start_time__lte=seg_end,
            ).select_related("user")

            for block in blocks:
                matched += 1
                if not _once(block.user, "block_soon", block.id, occ_date):
                    continue
                body = f"{block.title} à {block.start_time.strftime('%H:%M')}"
                if block.location:
                    body += f" ({block.location})"
                sent += send_to_user(block.user, f"Bientôt : {block.title}", body, url="/schedule")
                dispatch(block.user, "reminder.sent", {
                    "block": {
                        "id": block.id,
                        "title": block.title,
                        "start_time": block.start_time.strftime("%H:%M"),
                        "location": block.location,
                    },
                })

        # --- Pass 2: departure alerts for blocks tied to a place --------------
        prep_alerts = leave_alerts = 0
        placed_blocks = RecurringBlock.objects.filter(
            RecurringBlock.bounds_filter(now.date()),
            user_id__in=user_ids,
            active=True,
            day_of_week=now.weekday(),
            place__isnull=False,
        ).select_related("user", "user__profile", "place")

        for block in placed_blocks:
            travel = block.place.travel_minutes or 0
            if travel <= 0:
                continue
            profile = block.user.profile
            start_bmin = block.start_time.hour * 60 + block.start_time.minute
            w = commute_window(
                start_bmin,
                travel,
                profile.safety_margin_minutes or 0,
                profile.prep_time_minutes or 0,
            )

            # "Commence à te préparer" when prep-start falls in this window.
            if win_start_min <= w.unavailability_start <= end_min and _once(
                block.user, "block_prep", block.id, now.date()
            ):
                prep_alerts += send_to_user(
                    block.user,
                    "Prépare-toi",
                    f"Commence à te préparer pour {block.title}. "
                    f"Départ à {_fmt(w.departure)} ({travel} min de trajet).",
                    url="/schedule",
                )

            # "Pars maintenant" when the latest-departure time falls in the window.
            if win_start_min <= w.departure <= end_min and _once(
                block.user, "block_leave", block.id, now.date()
            ):
                leave_alerts += send_to_user(
                    block.user,
                    "Pars maintenant",
                    f"Pour {block.title} à {block.start_time.strftime('%H:%M')}, "
                    f"pars maintenant pour arriver à l'heure.",
                    url="/schedule",
                )

        # --- Pass 3: departure alerts for SCHEDULED TASKS tied to a place -----
        # Une tâche planifiée avec un lieu (ex: "Réunion 14h à l'UQAC") déclenche
        # les mêmes alertes de départ que les blocs récurrents (rappels géoloc).
        task_prep = task_leave = 0
        placed_tasks = ScheduledBlock.objects.filter(
            user_id__in=user_ids,
            date=now.date(),
            task__place__isnull=False,
        ).select_related("user", "user__profile", "task", "task__place")

        for sb in placed_tasks:
            travel = sb.task.place.travel_minutes or 0
            if travel <= 0:
                continue
            profile = sb.user.profile
            start_bmin = sb.start_time.hour * 60 + sb.start_time.minute
            w = commute_window(
                start_bmin,
                travel,
                profile.safety_margin_minutes or 0,
                profile.prep_time_minutes or 0,
            )
            label = sb.task.title
            place_name = sb.task.place.name

            if win_start_min <= w.unavailability_start <= end_min and _once(
                sb.user, "task_prep", sb.id, now.date()
            ):
                task_prep += send_to_user(
                    sb.user,
                    "Prépare-toi",
                    f"Commence à te préparer pour {label} ({place_name}). "
                    f"Départ à {_fmt(w.departure)} ({travel} min de trajet).",
                    url="/schedule",
                )

            if win_start_min <= w.departure <= end_min and _once(
                sb.user, "task_leave", sb.id, now.date()
            ):
                task_leave += send_to_user(
                    sb.user,
                    "Pars maintenant",
                    f"Pour {label} à {sb.start_time.strftime('%H:%M')} ({place_name}), "
                    f"pars maintenant pour arriver à l'heure.",
                    url="/schedule",
                )

        self.stdout.write(
            f"Reminders: {matched} bloc(s), {sent} push. "
            f"Départs blocs: {prep_alerts} prépa, {leave_alerts} départ. "
            f"Départs tâches: {task_prep} prépa, {task_leave} départ."
        )
