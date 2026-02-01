# Generated manually for user LLM preference

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_uploadeddocument_cache_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='preferred_llm',
            field=models.CharField(
                choices=[('gemini', 'Gemini (Google)'), ('claude', 'Claude (Anthropic)')],
                default='gemini',
                help_text='Modèle IA préféré pour le chat',
                max_length=20,
            ),
        ),
    ]
