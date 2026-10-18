from django.db import migrations, models


def copier_ancien_lien(apps, schema_editor):
    Message = apps.get_model('core', 'ConversationMessage')
    through = Message.attachments.through
    through.objects.bulk_create([
        through(conversationmessage_id=pk, uploadeddocument_id=doc_id)
        for pk, doc_id in Message.objects.exclude(attachment_id=None)
        .values_list('pk', 'attachment_id').iterator()
    ], ignore_conflicts=True)


class Migration(migrations.Migration):
    dependencies = [('core', '0044_conversation_epinglee')]

    operations = [
        migrations.AddField(
            model_name='uploadeddocument',
            name='text_preview',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='conversationmessage',
            name='attachments',
            field=models.ManyToManyField(blank=True, related_name='conversation_messages', to='core.uploadeddocument'),
        ),
        migrations.RunPython(copier_ancien_lien, migrations.RunPython.noop),
    ]
