"""
Send Web Push reminders + departure alerts for recurring blocks.

Run on a Railway cron at the same cadence as --lead (e.g. every 15 min with
--lead 15). Two passes:
  1. "Bientôt : <bloc>" when a block starts within --lead minutes.
  2. Departure alerts for blocks tied to a place (spec §9): "commence à te
     préparer" at début_indisponibilité, "pars maintenant" at l'heure limite de
     départ, computed from the commute engine (trajet + marge + préparation).

Degrades silently if push is not configured.
"""
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import PushSubscription, RecurringBlock, ScheduledBlock
from services.commute import commute_window
from services.push import push_configured, send_to_user


def _fmt(minutes: int) -> str:
    """Minutes-since-midnight -> 'HH:MM' (clamped to the day)."""
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class Command(BaseCommand):
    help = "Push reminders + departure alerts for blocks within --lead minutes."

    def add_arguments(self, parser):
        parser.add_argument("--lead", type=int, default=15, help="Minutes ahead to remind.")

    def handle(self, *args, **opts):
        if not push_configured():
            self.stdout.write("Web push not configured; nothing sent.")
            return

        lead = opts["lead"]
        now = timezone.localtime()
        now_min = now.hour * 60 + now.minute
        end_min = now_min + lead
        if end_min >= 24 * 60:
            # Window crosses midnight; skip this edge for the simple v1 cron.
            self.stdout.write("Reminder window crosses midnight; skipped.")
            return

        from services.webhooks import dispatch

        user_ids = list(
            PushSubscription.objects.values_list("user_id", flat=True).distinct()
        )
        start = now.time()
        end = (now + timedelta(minutes=lead)).time()

        # --- Pass 1: "Bientôt" reminders (block starting within the window) ----
        blocks = RecurringBlock.objects.filter(
            user_id__in=user_ids,
            active=True,
            day_of_week=now.weekday(),
            start_time__gte=start,
            start_time__lte=end,
        ).select_related("user")

        sent = 0
        for block in blocks:
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
            if now_min <= w.unavailability_start <= end_min:
                prep_alerts += send_to_user(
                    block.user,
                    "Prépare-toi",
                    f"Commence à te préparer pour {block.title}. "
                    f"Départ à {_fmt(w.departure)} ({travel} min de trajet).",
                    url="/schedule",
                )

            # "Pars maintenant" when the latest-departure time falls in the window.
            if now_min <= w.departure <= end_min:
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

            if now_min <= w.unavailability_start <= end_min:
                task_prep += send_to_user(
                    sb.user,
                    "Prépare-toi",
                    f"Commence à te préparer pour {label} ({place_name}). "
                    f"Départ à {_fmt(w.departure)} ({travel} min de trajet).",
                    url="/schedule",
                )

            if now_min <= w.departure <= end_min:
                task_leave += send_to_user(
                    sb.user,
                    "Pars maintenant",
                    f"Pour {label} à {sb.start_time.strftime('%H:%M')} ({place_name}), "
                    f"pars maintenant pour arriver à l'heure.",
                    url="/schedule",
                )

        self.stdout.write(
            f"Reminders: {blocks.count()} bloc(s), {sent} push. "
            f"Départs blocs: {prep_alerts} prépa, {leave_alerts} départ. "
            f"Départs tâches: {task_prep} prépa, {task_leave} départ."
        )
