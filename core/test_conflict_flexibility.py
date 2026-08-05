"""Le détecteur de conflits de /insights/conflicts/ (AIInsightsService.detect_conflicts)
doit suivre la MÊME règle que le moteur de placement et find_recurring_conflicts:
un bloc SOUPLE (flexible) qui chevauche un autre se replace tout seul -> PAS un
conflit. Seuls DEUX blocs FIXES en chevauchement sont un vrai conflit.

Avant le fix, il criait "4 conflits urgents" sur un planning de travailleur de
nuit pourtant sain (sommeil de récupération sur un cours, sommeils overnight
entre eux traités comme du "travail de nuit").
"""
import datetime as dt

import pytest
from django.contrib.auth.models import User

from core.models import RecurringBlock
from services.ai_insights import AIInsightsService


@pytest.fixture
def svc():
    return AIInsightsService()


def _rb(user, title, bt, dow, s, e, flex, night=False):
    return RecurringBlock.objects.create(
        user=user, title=title, block_type=bt, day_of_week=dow,
        start_time=dt.time.fromisoformat(s), end_time=dt.time.fromisoformat(e),
        flexibility=flex, is_night_shift=night,
    )


@pytest.mark.django_db
def test_flexible_sleep_over_fixed_course_is_not_flagged(svc):
    user = User.objects.create_user("cf_sleep_course", password="pw-123456")
    _rb(user, "Sommeil récupération", "sleep", 0, "07:00", "14:00", "flexible")
    _rb(user, "Statistiques", "course", 0, "09:00", "12:00", "fixed")
    conflicts = svc.detect_conflicts(user, days_ahead=7)
    assert not any("Sommeil" in c.message and "Statistiques" in c.message for c in conflicts), \
        [c.message for c in conflicts]


@pytest.mark.django_db
def test_two_fixed_courses_overlapping_is_still_a_conflict(svc):
    user = User.objects.create_user("cf_fixed_fixed", password="pw-123456")
    _rb(user, "Cours A", "course", 0, "09:00", "11:00", "fixed")
    _rb(user, "Cours B", "course", 0, "10:00", "12:00", "fixed")
    conflicts = svc.detect_conflicts(user, days_ahead=7)
    assert any(c.type == "overlap" for c in conflicts), [c.message for c in conflicts]


@pytest.mark.django_db
def test_night_worker_flexible_sleep_no_sleep_conflict(svc):
    user = User.objects.create_user("cf_night", password="pw-123456")
    _rb(user, "Travail de nuit", "work", 0, "22:00", "06:00", "fixed", night=True)
    _rb(user, "Sieste pré-quart", "sleep", 0, "14:00", "21:00", "flexible")
    # sommeil normal overnight (is_night_shift=True mais c'est du SOMMEIL, pas du travail)
    _rb(user, "Sommeil", "sleep", 4, "23:00", "07:00", "flexible", night=True)
    _rb(user, "Sommeil", "sleep", 5, "23:00", "07:00", "flexible", night=True)
    conflicts = svc.detect_conflicts(user, days_ahead=7)
    # ni sommeil/travail, ni sommeil/sommeil, ni le faux message "travail de nuit"
    assert not any("Sommeil" in c.message or "Sieste" in c.message for c in conflicts), \
        [c.message for c in conflicts]


@pytest.mark.django_db
def test_two_consecutive_night_shifts_are_not_flagged(svc):
    # Deux quarts de nuit CONSECUTIFS (Ven 19-07 + Sam 19-07) ne se chevauchent
    # PAS: Ven 19h->Sam 7h puis Sam 19h->Dim 7h, avec un trou Sam 7h-19h. Avant le
    # fix, le detecteur comptait la queue du matin du quart de samedi DEUX FOIS
    # (via son propre wrap overnight + via le quart de vendredi remonte le samedi)
    # -> faux "Chevauchement de 420min le samedi".
    user = User.objects.create_user("cf_two_nights", password="pw-123456")
    _rb(user, "Quart de nuit", "work", 4, "19:00", "07:00", "fixed", night=True)
    _rb(user, "Quart de nuit", "work", 5, "19:00", "07:00", "fixed", night=True)
    conflicts = svc.detect_conflicts(user, days_ahead=8)
    assert not any(c.type == "overlap" for c in conflicts), [c.message for c in conflicts]
