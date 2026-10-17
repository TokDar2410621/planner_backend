"""Rattache les messages existants a une conversation par defaut par utilisateur."""
from django.db import migrations


def rattacher(apps, schema_editor):
    Conversation = apps.get_model('core', 'Conversation')
    ConversationMessage = apps.get_model('core', 'ConversationMessage')
    User = apps.get_model('auth', 'User')
    for user in User.objects.all():
        orphelins = ConversationMessage.objects.filter(user=user, conversation__isnull=True)
        if not orphelins.exists():
            continue
        conv = Conversation.objects.create(user=user, titre="Conversation")
        orphelins.update(conversation=conv)


def detacher(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [('core', '0042_conversations')]
    operations = [migrations.RunPython(rattacher, detacher)]
