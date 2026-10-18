from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('core', '0043_rattacher_messages')]

    operations = [
        migrations.AddField(
            model_name='conversation',
            name='epinglee',
            field=models.BooleanField(default=False),
        ),
    ]
