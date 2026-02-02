"""
Chat Planning Functions - Planning-related functions for ChatOrchestrator.

Contains:
- Smart planning proposal generation
- Proposed blocks creation (sleep, meals)
- Document data extraction to planning
- Block overlap detection
"""
import logging
from datetime import time as dt_time, timedelta
from typing import Optional, List

from django.contrib.auth.models import User
from django.utils import timezone

logger = logging.getLogger(__name__)


def generate_smart_planning_proposal(user: User, overrides: dict = None) -> dict:
    """
    Analyze user's schedule and generate an intelligent planning proposal.
    """
    from core.models import RecurringBlock

    overrides = overrides or {}
    profile = user.profile
    blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')

    if not blocks.exists():
        return {
            'text': "Je n'ai pas encore assez d'informations sur ton emploi du temps. Envoie-moi ton planning de cours ou de travail!",
            'quick_replies': [
                {'label': "📚 Envoyer mon emploi du temps", 'value': 'upload'},
            ]
        }

    # Analyze schedule by day
    days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    schedule_by_day = {i: [] for i in range(7)}
    earliest_start_by_day = {}
    latest_end_by_day = {}

    for block in blocks:
        day = block.day_of_week
        schedule_by_day[day].append({
            'start': block.start_time,
            'end': block.end_time,
            'title': block.title,
            'type': block.block_type,
        })
        if day not in earliest_start_by_day or block.start_time < earliest_start_by_day[day]:
            earliest_start_by_day[day] = block.start_time
        if day not in latest_end_by_day or block.end_time > latest_end_by_day[day]:
            latest_end_by_day[day] = block.end_time

    # Find earliest start across weekdays
    earliest_weekday_start = None
    for day in range(5):
        if day in earliest_start_by_day:
            if earliest_weekday_start is None or earliest_start_by_day[day] < earliest_weekday_start:
                earliest_weekday_start = earliest_start_by_day[day]

    # Calculate proposed sleep schedule
    min_sleep = profile.min_sleep_hours or 7
    transport_time = profile.transport_time_minutes or 30

    if earliest_weekday_start:
        wake_minutes = earliest_weekday_start.hour * 60 + earliest_weekday_start.minute
        wake_minutes -= transport_time + 60
        wake_hour = max(5, wake_minutes // 60)
        wake_minute = wake_minutes % 60 if wake_minutes > 0 else 0
        bed_hour = (wake_hour - min_sleep) % 24
        if bed_hour < 12:
            bed_hour += 24 - min_sleep
        proposed_wake = f"{wake_hour:02d}:{wake_minute:02d}"
        proposed_bed = f"{bed_hour % 24:02d}:00"
    else:
        proposed_wake = "07:00"
        proposed_bed = "23:00"

    # Apply overrides
    if overrides.get('bedtime'):
        proposed_bed = overrides['bedtime']
    if overrides.get('wake_time'):
        proposed_wake = overrides['wake_time']

    # Find free slots
    def find_free_slots(day_blocks, day_start=dt_time(7, 0), day_end=dt_time(23, 0)):
        if not day_blocks:
            return [(day_start, day_end)]
        sorted_blocks = sorted(day_blocks, key=lambda x: x['start'])
        free_slots = []
        current_time = day_start
        for block in sorted_blocks:
            if block['start'] > current_time:
                free_slots.append((current_time, block['start']))
            current_time = max(current_time, block['end'])
        if current_time < day_end:
            free_slots.append((current_time, day_end))
        return free_slots

    # Build the proposal
    proposal_lines = ["**Voici ma proposition de planning optimisé:**\n"]
    proposal_lines.append(f"**🛏️ Sommeil ({min_sleep}h minimum)**")
    proposal_lines.append(f"  Coucher: {proposed_bed} → Réveil: {proposed_wake}")
    proposal_lines.append("")

    # Calculate available study time
    total_available_minutes = 0
    peak_available_minutes = 0
    peak_time = profile.peak_productivity_time or 'morning'
    all_study_slots = []

    for day in range(7):
        day_blocks = schedule_by_day.get(day, [])
        free_slots = find_free_slots(day_blocks)

        for slot_start, slot_end in free_slots:
            duration = (slot_end.hour * 60 + slot_end.minute) - (slot_start.hour * 60 + slot_start.minute)
            slot_start_minutes = slot_start.hour * 60 + slot_start.minute

            # Skip meal times
            is_meal_time = (420 <= slot_start_minutes < 510) or \
                          (720 <= slot_start_minutes < 810) or \
                          (1110 <= slot_start_minutes < 1200)

            if duration >= 60 and not is_meal_time:
                total_available_minutes += duration

                is_peak = False
                if peak_time == 'morning' and slot_start.hour < 12:
                    is_peak = True
                elif peak_time == 'afternoon' and 12 <= slot_start.hour < 18:
                    is_peak = True
                elif peak_time == 'evening' and slot_start.hour >= 18:
                    is_peak = True

                if is_peak:
                    peak_available_minutes += duration

                all_study_slots.append({
                    'day': day,
                    'day_name': days_fr[day],
                    'start': slot_start,
                    'end': slot_end,
                    'duration': duration,
                    'is_peak': is_peak
                })

    proposal_lines.append(f"**📚 Travail personnel (pic: {profile.get_peak_productivity_time_display()})**")
    proposal_lines.append(f"  📊 Temps disponible/semaine: ~{total_available_minutes // 60}h")

    if peak_available_minutes > 0:
        proposal_lines.append(f"  🎯 Créneaux optimaux: ~{peak_available_minutes // 60}h")

    proposal_lines.append("")

    # Show top study slots
    if all_study_slots:
        # Sort by peak first, then by duration
        all_study_slots.sort(key=lambda x: (-x['is_peak'], -x['duration']))

        proposal_lines.append("  **💻 Meilleurs créneaux:**")
        for slot in all_study_slots[:5]:
            hours = slot['duration'] // 60
            mins = slot['duration'] % 60
            dur_str = f"{hours}h{mins:02d}" if mins else f"{hours}h"
            peak_marker = "🎯" if slot['is_peak'] else "  "
            proposal_lines.append(
                f"    {peak_marker} {slot['day_name']}: "
                f"{slot['start'].hour:02d}:{slot['start'].minute:02d} - "
                f"{slot['end'].hour:02d}:{slot['end'].minute:02d} ({dur_str})"
            )
        proposal_lines.append("")

    proposal_lines.append("**Cette proposition est basée sur ton emploi du temps actuel.**")
    proposal_lines.append("Je peux créer ces blocs automatiquement ou les ajuster.")

    return {
        'text': "\n".join(proposal_lines),
        'quick_replies': [
            {'label': "✅ Crée ce planning!", 'value': 'Oui, crée ce planning!'},
            {'label': "🔧 Ajuster", 'value': 'Je voudrais ajuster quelque chose'},
            {'label': "💬 J'ai des questions", 'value': "J'ai des questions sur cette proposition"},
        ],
        'proposed_schedule': {
            'sleep': {'bedtime': proposed_bed, 'wake': proposed_wake},
            'study_slots': all_study_slots[:6],
        }
    }


def create_proposed_blocks(user: User) -> List[dict]:
    """
    Create the proposed blocks (sleep, meals) in the database.
    """
    from core.models import RecurringBlock

    profile = user.profile
    created = []
    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    # Get existing blocks to find earliest start
    blocks = RecurringBlock.objects.filter(user=user, active=True)
    schedule_by_day = {i: [] for i in range(7)}
    earliest_weekday_start = None

    for block in blocks:
        schedule_by_day[block.day_of_week].append({
            'start': block.start_time,
            'end': block.end_time,
        })
        if block.day_of_week < 5:
            if earliest_weekday_start is None or block.start_time < earliest_weekday_start:
                earliest_weekday_start = block.start_time

    # Calculate sleep times
    min_sleep = profile.min_sleep_hours or 7
    transport_time = profile.transport_time_minutes or 30

    if earliest_weekday_start:
        wake_minutes = earliest_weekday_start.hour * 60 + earliest_weekday_start.minute
        wake_minutes -= transport_time + 60
        wake_hour = max(5, wake_minutes // 60)
        wake_minute = wake_minutes % 60 if wake_minutes > 0 else 0
        bed_hour = (wake_hour - min_sleep) % 24
        if bed_hour < 12:
            bed_hour = 24 - min_sleep + wake_hour
            bed_hour = bed_hour % 24
    else:
        wake_hour, wake_minute = 7, 0
        bed_hour = 23

    wake_time = dt_time(wake_hour, wake_minute)
    bed_time = dt_time(bed_hour % 24, 0)

    # Create sleep blocks for each day
    for day in range(7):
        existing = RecurringBlock.objects.filter(
            user=user, day_of_week=day, block_type='sleep'
        ).exists()

        if not existing:
            RecurringBlock.objects.create(
                user=user,
                title="Sommeil",
                block_type='sleep',
                day_of_week=day,
                start_time=bed_time,
                end_time=wake_time,
            )
            created.append({'title': f'Sommeil ({day_names[day]})'})

    # Helper to find free slots
    def find_free_slots(day_blocks, day_start=dt_time(7, 0), day_end=dt_time(23, 0)):
        if not day_blocks:
            return [(day_start, day_end)]
        sorted_blocks = sorted(day_blocks, key=lambda x: x['start'])
        free_slots = []
        current_time = day_start
        for block in sorted_blocks:
            if block['start'] > current_time:
                free_slots.append((current_time, block['start']))
            if block['end'] > current_time:
                current_time = block['end']
        if current_time < day_end:
            free_slots.append((current_time, day_end))
        return free_slots

    # Create meal blocks for weekdays
    for day in range(5):
        existing_meals = RecurringBlock.objects.filter(
            user=user, day_of_week=day, block_type='meal'
        ).values_list('title', flat=True)
        existing_meals = list(existing_meals)

        free_slots = find_free_slots(schedule_by_day[day])

        for slot_start, slot_end in free_slots:
            slot_start_minutes = slot_start.hour * 60 + slot_start.minute
            slot_end_minutes = slot_end.hour * 60 + slot_end.minute
            duration = slot_end_minutes - slot_start_minutes

            # Breakfast (6h-9h)
            if 'Petit-déjeuner' not in existing_meals and 360 <= slot_start_minutes < 540 and duration >= 30:
                end_minute = slot_start.minute + 30
                end_hour = slot_start.hour + (end_minute // 60)
                end_minute = end_minute % 60
                RecurringBlock.objects.create(
                    user=user,
                    title='Petit-déjeuner',
                    block_type='meal',
                    day_of_week=day,
                    start_time=slot_start,
                    end_time=dt_time(end_hour, end_minute),
                )
                created.append({'title': f'Petit-déjeuner ({day_names[day]})'})
                existing_meals.append('Petit-déjeuner')

            # Lunch (11h30-14h)
            elif 'Déjeuner' not in existing_meals and 690 <= slot_start_minutes < 840 and duration >= 45:
                end_minute = slot_start.minute + 45
                end_hour = slot_start.hour + (end_minute // 60)
                end_minute = end_minute % 60
                RecurringBlock.objects.create(
                    user=user,
                    title='Déjeuner',
                    block_type='meal',
                    day_of_week=day,
                    start_time=slot_start,
                    end_time=dt_time(end_hour, end_minute),
                )
                created.append({'title': f'Déjeuner ({day_names[day]})'})
                existing_meals.append('Déjeuner')

            # Dinner (18h-21h)
            elif 'Dîner' not in existing_meals and 1080 <= slot_start_minutes < 1260 and duration >= 45:
                end_minute = slot_start.minute + 45
                end_hour = slot_start.hour + (end_minute // 60)
                end_minute = end_minute % 60
                RecurringBlock.objects.create(
                    user=user,
                    title='Dîner',
                    block_type='meal',
                    day_of_week=day,
                    start_time=slot_start,
                    end_time=dt_time(end_hour, end_minute),
                )
                created.append({'title': f'Dîner ({day_names[day]})'})
                existing_meals.append('Dîner')

    return created


def add_extracted_to_planning(user: User) -> dict:
    """
    Add extracted data from the most recent document to the planning.
    """
    from services.document_processor import DocumentProcessor
    from core.models import RecurringBlock, UploadedDocument

    logger.info(f"add_extracted_to_planning called for user {user.id}")

    # Check for processing document
    ten_minutes_ago = timezone.now() - timedelta(minutes=10)
    processing_doc = UploadedDocument.objects.filter(
        user=user,
        processed=False,
        uploaded_at__gte=ten_minutes_ago
    ).order_by('-uploaded_at').first()

    if processing_doc:
        logger.info(f"Found processing document {processing_doc.id}, uploaded at {processing_doc.uploaded_at}")
        return {
            'text': "Le document est encore en cours de traitement. Patiente quelques secondes!",
            'quick_replies': [
                {'label': "🔄 Vérifier à nouveau", 'value': "Tu as fini d'analyser mon document?"},
            ]
        }

    # Get the most recent processed document
    recent_doc = UploadedDocument.objects.filter(
        user=user,
        processed=True
    ).order_by('-uploaded_at').first()

    logger.info(f"Recent processed doc: {recent_doc.id if recent_doc else 'None'}")
    if recent_doc:
        logger.info(f"Doc extracted_data: {bool(recent_doc.extracted_data)}, keys: {list(recent_doc.extracted_data.keys()) if recent_doc.extracted_data else []}")
        if recent_doc.extracted_data:
            courses = recent_doc.extracted_data.get('courses', [])
            shifts = recent_doc.extracted_data.get('shifts', [])
            logger.info(f"Extracted: {len(courses)} courses, {len(shifts)} shifts")

    if not recent_doc or not recent_doc.extracted_data:
        logger.warning(f"No recent doc with extracted_data for user {user.id}")
        return {
            'text': "Je n'ai pas trouvé de document récent avec des données à ajouter. Envoie-moi ton emploi du temps!",
            'quick_replies': [
                {'label': "📚 Envoyer mon emploi du temps", 'value': 'upload'},
            ]
        }

    # Check if blocks already exist for this document
    existing_blocks = RecurringBlock.objects.filter(source_document=recent_doc).count()
    logger.info(f"Existing blocks for doc {recent_doc.id}: {existing_blocks}")
    if existing_blocks > 0:
        return {
            'text': f"Les données de ce document ont déjà été ajoutées ({existing_blocks} blocs créés).",
            'quick_replies': [
                {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                {'label': "📊 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
            ]
        }

    # Create blocks from extracted data
    try:
        logger.info(f"Creating blocks from extracted_data: {recent_doc.extracted_data}")
        processor = DocumentProcessor()
        created_blocks = processor._create_recurring_blocks(recent_doc, recent_doc.extracted_data)
        logger.info(f"Created {len(created_blocks)} blocks")

        if created_blocks:
            block_summary = []
            courses = [b for b in created_blocks if b.block_type == 'course']
            shifts = [b for b in created_blocks if b.block_type == 'work']
            others = [b for b in created_blocks if b.block_type not in ['course', 'work']]

            if courses:
                block_summary.append(f"📚 {len(courses)} cours")
            if shifts:
                block_summary.append(f"💼 {len(shifts)} créneaux de travail")
            if others:
                block_summary.append(f"📅 {len(others)} autres")

            summary = ", ".join(block_summary) if block_summary else f"{len(created_blocks)} blocs"

            return {
                'text': f"J'ai ajouté {len(created_blocks)} blocs à ton planning: {summary}\n\nJe peux maintenant te proposer un planning optimisé!",
                'quick_replies': [
                    {'label': "📊 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                ]
            }
        else:
            return {
                'text': "Je n'ai pas pu créer de blocs à partir des données extraites. Les jours ou heures ne sont peut-être pas au bon format.",
                'quick_replies': [
                    {'label': "📝 Décrire mon emploi du temps", 'value': "Je vais te décrire mon emploi du temps"},
                ]
            }
    except Exception as e:
        logger.error(f"Error adding extracted data: {e}")
        return {
            'text': "Une erreur s'est produite lors de l'ajout des données.",
            'quick_replies': [
                {'label': "📝 Décrire mon emploi du temps", 'value': "Je vais te décrire mon emploi du temps"},
            ]
        }


def check_block_overlap(user: User, day: int, start_time, end_time) -> tuple:
    """
    Check if a block would overlap with existing blocks.

    Args:
        user: The user
        day: Day of week (0-6)
        start_time: Start time as string "HH:MM" or datetime.time object
        end_time: End time as string "HH:MM" or datetime.time object

    Returns:
        tuple: (has_overlap: bool, overlapping_block_title: str or None)
    """
    from core.models import RecurringBlock

    # Convert string times to time objects if necessary
    def parse_time(t):
        if isinstance(t, dt_time):
            return t
        if isinstance(t, str):
            try:
                parts = t.split(':')
                return dt_time(int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                return None
        return None

    new_start = parse_time(start_time)
    new_end = parse_time(end_time)

    if new_start is None or new_end is None:
        logger.warning(f"Invalid time format: start={start_time}, end={end_time}")
        return False, None  # Can't check, let it through

    existing_blocks = RecurringBlock.objects.filter(user=user, day_of_week=day, active=True)

    for existing in existing_blocks:
        existing_start = existing.start_time
        existing_end = existing.end_time

        # Handle overnight blocks (end < start means it goes past midnight)
        if existing_end <= existing_start:
            # Overnight existing block: overlaps if new block starts after existing starts
            # or if new block ends before existing ends (on the next day part)
            if new_start >= existing_start or new_end <= existing_end:
                logger.info(f"Overlap detected with overnight block '{existing.title}'")
                return True, existing.title
        elif new_end <= new_start:
            # New block is overnight
            if existing_start >= new_start or existing_end <= new_end:
                logger.info(f"Overlap detected: new overnight block overlaps with '{existing.title}'")
                return True, existing.title
        else:
            # Normal case: both blocks are within same day
            # Two ranges overlap if: start1 < end2 AND start2 < end1
            if new_start < existing_end and existing_start < new_end:
                logger.info(f"Overlap detected with '{existing.title}' ({existing_start}-{existing_end})")
                return True, existing.title

    return False, None
