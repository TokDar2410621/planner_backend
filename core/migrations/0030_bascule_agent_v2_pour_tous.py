# Bascule generale vers l'agent v2, decidee le 2026-08-30 sur la serie
# decisive du banc: verite 20/20 aux trois passages pour v2, pendant que v1
# tombait a 18,8 puis 17,6 en annoncant des suppressions jamais executees
# (docs/decision-bascule-agent-v2-2026-08-30.md).
#
# Deux effets, dans l'ordre:
# - les profils existants passent a agent_v2=True (UPDATE en masse);
# - le defaut du champ passe a True: un nouveau compte nait sur v2.
#
# Le retour arriere reste ce qu'il a toujours ete: un UPDATE par compte,
# sans deploiement. Le reverse de cette migration ne remet volontairement
# PERSONNE sur v1: annuler un deploiement ne doit pas changer l'agent de
# 185 comptes en silence.
from django.db import migrations, models


def _basculer_tout_le_monde(apps, schema_editor):
    UserProfile = apps.get_model("core", "UserProfile")
    combien = UserProfile.objects.filter(agent_v2=False).update(agent_v2=True)
    total = UserProfile.objects.count()
    print(f"bascule agent_v2: {combien} profil(s) bascule(s), {total} au total, tous sur v2")


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0029_agent_v2"),
    ]

    operations = [
        migrations.RunPython(_basculer_tout_le_monde, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="userprofile",
            name="agent_v2",
            field=models.BooleanField(default=True),
        ),
    ]
