"""
iCalendar (.ics) feed generation for Planner AI.

Turns a user's schedule into an RFC 5545 VCALENDAR that Google Calendar,
Apple Calendar, Outlook, etc. can *subscribe* to (auto-refreshing). This is the
"emitter" side of the life-assistant layer: Planner blocks show up natively in
whatever calendar the student already lives in.

Design choices:
- Recurring blocks (courses/work/sleep) are emitted as weekly VEVENTs with an
  RRULE, in *floating* local time (no TZID). RecurringBlock.start_time /
  end_time are naive TimeFields (wall-clock "09:00"), so floating time renders
  at 09:00 in the subscriber's own calendar timezone. This sidesteps the
  server TZ (Europe/Paris) vs. student TZ (America/Toronto) mismatch and is
  DST-safe for fixed wall-clock class times.
- Task deadlines are precise instants (aware, stored UTC) → emitted in UTC (Z).
- No external dependency: hand-rolled with correct escaping, CRLF, and 75-octet
  line folding so strict clients (Google) accept the feed.
"""
from __future__ import annotations

from datetime import date, timedelta
from datetime import timezone as dt_timezone

from django.utils import timezone

# day_of_week: 0 = Monday (matches Python date.weekday() and RecurringBlock)
_BYDAY = {0: "MO", 1: "TU", 2: "WE", 3: "TH", 4: "FR", 5: "SA", 6: "SU"}

_BLOCK_TYPE_LABELS = {
    "course": "Cours",
    "work": "Travail",
    "sleep": "Sommeil",
    "meal": "Repas",
    "sport": "Sport",
    "project": "Projet",
    "revision": "Révision",
    "other": "Autre",
}


def _esc(value) -> str:
    """Escape a TEXT value per RFC 5545 (backslash, semicolon, comma, newline)."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\")
    s = s.replace(";", "\\;")
    s = s.replace(",", "\\,")
    s = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return s


def _fold(line: str) -> str:
    """Fold a content line to <=75 octets, continuation lines start with a space."""
    encoded = line.encode("utf-8")
    if len(encoded) <= 75:
        return line
    out = []
    chunk = b""
    for ch in line:
        b = ch.encode("utf-8")
        # 75 octets max per line; continuation lines carry a leading space,
        # so subsequent chunks get 74 octets of payload.
        limit = 75 if not out else 74
        if len(chunk) + len(b) > limit:
            out.append(chunk.decode("utf-8"))
            chunk = b
        else:
            chunk += b
    out.append(chunk.decode("utf-8"))
    return "\r\n ".join(out)


def _dt_utc(dt) -> str:
    """Format an aware datetime as an iCal UTC stamp: YYYYMMDDTHHMMSSZ."""
    return timezone.localtime(dt, dt_timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dt_floating(d: date, t) -> str:
    """Format a naive date+time as floating local: YYYYMMDDTHHMMSS (no TZID/Z)."""
    return f"{d.strftime('%Y%m%d')}T{t.strftime('%H%M%S')}"


def _next_weekday_on_or_after(anchor: date, weekday: int) -> date:
    """First date >= anchor that falls on `weekday` (0=Monday)."""
    delta = (weekday - anchor.weekday()) % 7
    return anchor + timedelta(days=delta)


def build_calendar(user, include_tasks: bool = False, cal_name: str | None = None) -> str:
    """Return the full .ics document (str) for a user's schedule."""
    # Local import keeps this module importable without Django app-loading order
    # concerns during migrations.
    from core.models import RecurringBlock, ScheduledBlock, Task

    now = timezone.now()
    dtstamp = _dt_utc(now)
    today = timezone.localtime(now).date()
    owner = user.first_name or user.username
    name = cal_name or f"Planner AI — {owner}"

    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Planner AI//Calendar Feed//FR",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_esc(name)}",
        "X-WR-CALDESC:Horaire et tâches synchronisés depuis Planner AI",
        "REFRESH-INTERVAL;VALUE=DURATION:PT1H",
        "X-PUBLISHED-TTL:PT1H",
    ]

    # --- Recurring blocks -> weekly recurring VEVENTs (floating local time) ---
    blocks = RecurringBlock.objects.filter(user=user, active=True).order_by(
        "day_of_week", "start_time"
    )
    for b in blocks:
        start_date = _next_weekday_on_or_after(today, b.day_of_week)
        end_date = start_date
        # Overnight block (e.g. night shift 22:00 -> 06:00): DTEND rolls to +1 day.
        if b.end_time <= b.start_time:
            end_date = start_date + timedelta(days=1)
        label = _BLOCK_TYPE_LABELS.get(b.block_type, b.block_type)
        lines += [
            "BEGIN:VEVENT",
            f"UID:recblock-{b.id}@planner-ai",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART:{_dt_floating(start_date, b.start_time)}",
            f"DTEND:{_dt_floating(end_date, b.end_time)}",
            f"RRULE:FREQ=WEEKLY;BYDAY={_BYDAY[b.day_of_week]}",
            f"SUMMARY:{_esc(b.title)}",
            f"CATEGORIES:{_esc(label)}",
        ]
        if b.location:
            lines.append(f"LOCATION:{_esc(b.location)}")
        lines.append("END:VEVENT")

    if include_tasks:
        # --- Scheduled task blocks (next 30 days) -> dated VEVENTs ------------
        horizon = today + timedelta(days=30)
        sched = (
            ScheduledBlock.objects.filter(
                user=user, date__gte=today, date__lt=horizon
            )
            .select_related("task")
            .order_by("date", "start_time")
        )
        for s in sched:
            end_date = s.date if s.end_time > s.start_time else s.date + timedelta(days=1)
            lines += [
                "BEGIN:VEVENT",
                f"UID:schedblock-{s.id}@planner-ai",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{_dt_floating(s.date, s.start_time)}",
                f"DTEND:{_dt_floating(end_date, s.end_time)}",
                f"SUMMARY:{_esc(s.task.title)}",
                f"STATUS:{'CONFIRMED' if not s.actually_completed else 'CONFIRMED'}",
                "CATEGORIES:Tâche",
                "END:VEVENT",
            ]

        # --- Task deadlines (precise instants, UTC) -> marker VEVENTs ---------
        deadlines = Task.objects.filter(
            user=user, completed=False, deadline__isnull=False, deadline__gte=now
        ).order_by("deadline")
        for t in deadlines:
            start = _dt_utc(t.deadline)
            end = _dt_utc(t.deadline + timedelta(minutes=30))
            lines += [
                "BEGIN:VEVENT",
                f"UID:task-deadline-{t.id}@planner-ai",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{start}",
                f"DTEND:{end}",
                f"SUMMARY:{_esc('⏰ ' + t.title + ' (échéance)')}",
                "CATEGORIES:Échéance",
                "END:VEVENT",
            ]

    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(l) for l in lines) + "\r\n"
