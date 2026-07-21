"""
Send Web Push reminders for recurring blocks starting soon.

Run on a Railway cron at the same cadence as --lead (e.g. every 15 min with
--lead 15) so each block is reminded once. Degrades silently if push is not
configured. This is the proactive "effector" of the life-assistant layer.
"""
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import PushSubscription, RecurringBlock
from services.push import push_configured, send_to_user


class Command(BaseCommand):
    help = "Send push reminders for recurring blocks starting within --lead minutes."

    def add_arguments(self, parser):
        parser.add_argument("--lead", type=int, default=15, help="Minutes ahead to remind.")

    def handle(self, *args, **opts):
        if not push_configured():
            self.stdout.write("Web push not configured; nothing sent.")
            return

        lead = opts["lead"]
        now = timezone.localtime()
        start = now.time()
        end = (now + timedelta(minutes=lead)).time()
        if start > end:
            # Window crosses midnight; skip this edge for the simple v1 cron.
            self.stdout.write("Reminder window crosses midnight; skipped.")
            return

        user_ids = PushSubscription.objects.values_list("user_id", flat=True).distinct()
        blocks = RecurringBlock.objects.filter(
            user_id__in=list(user_ids),
            active=True,
            day_of_week=now.weekday(),  # 0 = Monday
            start_time__gte=start,
            start_time__lte=end,
        ).select_related("user")

        sent = 0
        for block in blocks:
            body = f"{block.title} à {block.start_time.strftime('%H:%M')}"
            if block.location:
                body += f" ({block.location})"
            sent += send_to_user(block.user, f"Bientôt : {block.title}", body, url="/schedule")

        self.stdout.write(f"Reminders: {blocks.count()} block(s), {sent} push(es) sent.")
