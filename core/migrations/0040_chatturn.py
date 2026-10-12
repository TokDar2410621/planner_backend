from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0039_merge_0037_memoire_preferences_0038_auth_anonyme'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]
    operations = [migrations.CreateModel(
        name='ChatTurn',
        fields=[
            ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ('turn_id', models.UUIDField()),
            ('payload_hash', models.CharField(max_length=64)),
            ('status', models.CharField(choices=[('accepted', 'Accepté'), ('running', 'En cours'), ('completed', 'Terminé'), ('failed', 'Échec'), ('interrupted', 'Interrompu')], default='accepted', max_length=16)),
            ('result', models.JSONField(blank=True, default=dict)),
            ('http_status', models.PositiveSmallIntegerField(default=200)),
            ('created_at', models.DateTimeField(auto_now_add=True)),
            ('updated_at', models.DateTimeField(auto_now=True)),
            ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_turns', to=settings.AUTH_USER_MODEL)),
        ],
        options={'constraints': [models.UniqueConstraint(fields=('user', 'turn_id'), name='unique_user_chat_turn')]},
    )]
