"""Conversations : un fil d'historique par conversation, style OpenCode."""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import Conversation, ConversationMessage


class ConversationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='darius', password='x')
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_creer_et_lister(self):
        r = self.client.post('/api/conversations/', {'titre': 'Sport'})
        self.assertEqual(r.status_code, 201)
        conv_id = r.json()['id']
        r = self.client.get('/api/conversations/')
        self.assertEqual(r.status_code, 200)
        ids = [c['id'] for c in r.json()]
        self.assertIn(conv_id, ids)

    def test_messages_d_une_conversation(self):
        conv = Conversation.objects.create(user=self.user, titre='T1')
        ConversationMessage.objects.create(
            user=self.user, conversation=conv, role='user', content='salut')
        r = self.client.get(f'/api/conversations/{conv.pk}/messages/')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()), {'conversation', 'messages'})
        self.assertEqual(r.json()['conversation']['id'], conv.pk)
        msgs = r.json()['messages']
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]['content'], 'salut')

    def test_archiver(self):
        conv = Conversation.objects.create(user=self.user, titre='T1')
        r = self.client.patch(f'/api/conversations/{conv.pk}/', {'archivee': True},
                              format='json')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['archivee'])
        r = self.client.get('/api/conversations/')
        self.assertNotIn(conv.pk, [c['id'] for c in r.json()])

    def test_isolation_entre_utilisateurs(self):
        autre = User.objects.create_user(username='autre', password='x')
        conv = Conversation.objects.create(user=autre, titre='Prive')
        r = self.client.get(f'/api/conversations/{conv.pk}/messages/')
        self.assertEqual(r.status_code, 404)

    def test_renommer(self):
        conv = Conversation.objects.create(user=self.user, titre='Ancien')
        r = self.client.patch(f'/api/conversations/{conv.pk}/', {'titre': 'Nouveau'},
                              format='json')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['titre'], 'Nouveau')

    def test_epingler_et_trier(self):
        ancien = Conversation.objects.create(user=self.user, titre='Ancien')
        Conversation.objects.create(user=self.user, titre='Recent')
        r = self.client.patch(f'/api/conversations/{ancien.pk}/', {'epinglee': True}, format='json')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['epinglee'])
        liste = self.client.get('/api/conversations/').json()
        self.assertEqual(liste[0]['id'], ancien.pk)
        self.assertTrue(liste[0]['epinglee'])

    def test_supprimer_efface_le_fil_et_ses_messages(self):
        conv = Conversation.objects.create(user=self.user, titre='À supprimer')
        ConversationMessage.objects.create(user=self.user, conversation=conv, role='user', content='privé')
        r = self.client.delete(f'/api/conversations/{conv.pk}/')
        self.assertEqual(r.status_code, 204)
        self.assertFalse(Conversation.objects.filter(pk=conv.pk).exists())
        self.assertFalse(ConversationMessage.objects.filter(conversation_id=conv.pk).exists())
        self.assertEqual(self.client.get(f'/api/conversations/{conv.pk}/messages/').status_code, 404)
