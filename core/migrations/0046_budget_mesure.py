from django.db import migrations, models
from django.utils import timezone
from datetime import datetime, time, timedelta


def reprendre_mesures(apps, schema_editor):
    Budget = apps.get_model('core', 'BudgetJetonsJournalier')
    Message = apps.get_model('core', 'ConversationMessage')
    jour = timezone.localdate()
    debut = timezone.make_aware(datetime.combine(jour, time.min), timezone.get_current_timezone())
    fin = timezone.make_aware(datetime.combine(jour + timedelta(days=1), time.min), timezone.get_current_timezone())
    for budget in Budget.objects.filter(jour=jour).iterator():
        mesures = 0
        inconnus = 0
        messages = Message.objects.filter(
            user_id=budget.user_id, role='assistant',
            created_at__gte=debut, created_at__lt=fin,
        ).values_list('metadata', flat=True)
        for metadata in messages.iterator():
            usages = metadata.get('usage', []) if isinstance(metadata, dict) else []
            if not isinstance(usages, list):
                continue
            for usage in usages:
                if not isinstance(usage, dict):
                    continue
                total = usage.get('total')
                if usage.get('usage_connu') is True and type(total) is int and total > 0:
                    mesures += total
                elif usage.get('usage_connu') is False and usage.get('succes') is True:
                    inconnus += 1
        Budget.objects.filter(pk=budget.pk).update(
            jetons_mesures=mesures, appels_non_mesures=inconnus,
            mesure_partielle=True,
        )


class Migration(migrations.Migration):
    dependencies = [('core', '0045_conversationmessage_attachments')]

    operations = [
        migrations.AddField(
            model_name='budgetjetonsjournalier', name='jetons_mesures',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='budgetjetonsjournalier', name='appels_non_mesures',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='budgetjetonsjournalier', name='mesure_partielle',
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(reprendre_mesures, migrations.RunPython.noop),
    ]
