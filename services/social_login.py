"""
Shared social-login account resolution (Google, Apple, ...).

A provider-verified email has a single real owner, so several local accounts on
that email are the same person's duplicates: sign into the one that holds their
data (most recurring blocks, then most recent), never hard-block. New accounts
get a unique username derived from the email. Used by both /auth/google/ and
/auth/apple/ so the two never drift.
"""
from django.contrib.auth.models import User
from django.db.models import Count


def resolve_social_user(email: str, first_name: str = '', last_name: str = ''):
    """Return (user, created) for a provider-verified email."""
    matching = list(
        User.objects.filter(email__iexact=email)
        .annotate(_nblocks=Count('recurring_blocks'))
        .order_by('-_nblocks', '-last_login', '-date_joined')
    )
    if matching:
        user = matching[0]
        # Backfill a missing name from the provider (first sign-in via that app).
        changed = []
        if not user.first_name and first_name:
            user.first_name = first_name[:150]
            changed.append('first_name')
        if not user.last_name and last_name:
            user.last_name = last_name[:150]
            changed.append('last_name')
        if changed:
            user.save(update_fields=changed)
        return user, False

    base = (email.split('@')[0] or 'user')[:140]
    username = base
    i = 1
    while User.objects.filter(username=username).exists():
        username = f"{base}{i}"[:150]
        i += 1
    user = User.objects.create(
        username=username, email=email,
        first_name=(first_name or '')[:150], last_name=(last_name or '')[:150],
    )
    return user, True
