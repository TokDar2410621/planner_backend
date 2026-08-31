from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        # Reveil silencieux iOS: signaux sur les modeles du planning.
        from core import signaux_reveil

        signaux_reveil.connecter()
