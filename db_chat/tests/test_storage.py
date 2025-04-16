import uuid
from unittest.mock import MagicMock, patch

import pytest

from db_chat import storage as chat_storage


@pytest.fixture
def conversation_id():
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        REDIS_URL = "redis://localhost:6379/0"

    monkeypatch.setattr("db_chat.storage.settings", DummySettings)
    yield


class TestInMemoryConversationStorage:
    def setup_method(self):
        self.storage = chat_storage.InMemoryConversationStorage()

    def test_create_and_get_conversation(self, conversation_id):
        cid = self.storage.create_conversation()
        assert isinstance(cid, str)
        assert self.storage.get_conversation(cid) == []

    def test_save_and_get_message(self, conversation_id):
        cid = self.storage.create_conversation()
        msg = {"role": "user", "content": "hi"}
        assert self.storage.save_message(cid, msg)
        conv = self.storage.get_conversation(cid)
        assert len(conv) == 1
        assert conv[0]["role"] == "user"

    def test_delete_conversation(self, conversation_id):
        cid = self.storage.create_conversation()
        self.storage.save_message(cid, {"role": "user", "content": "hi"})
        assert self.storage.delete_conversation(cid)
        assert self.storage.get_conversation(cid) == []

    def test_update_expiry_noop(self, conversation_id):
        cid = self.storage.create_conversation()
        assert self.storage.update_expiry(cid)
        assert not self.storage.update_expiry("nonexistent")


class TestRedisConversationStorage:
    @patch("db_chat.storage.redis.Redis")
    def test_create_and_get_conversation(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        storage = chat_storage.RedisConversationStorage()
        storage.redis_client = mock_client
        cid = storage.create_conversation()
        assert isinstance(cid, str)

    @patch("db_chat.storage.redis.Redis")
    def test_save_and_get_message(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        storage = chat_storage.RedisConversationStorage()
        storage.redis_client = mock_client
        cid = "test-cid"
        msg = {"role": "user", "content": "hi"}
        mock_client.exists.return_value = True
        mock_client.lrange.return_value = [b'{"role": "user", "content": "hi"}']
        assert storage.save_message(cid, msg)
        conv = storage.get_conversation(cid)
        assert isinstance(conv, list)
        assert conv[0]["role"] == "user"

    @patch("db_chat.storage.redis.Redis")
    def test_delete_conversation(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        storage = chat_storage.RedisConversationStorage()
        storage.redis_client = mock_client
        cid = "test-cid"
        assert storage.delete_conversation(cid)

    @patch("db_chat.storage.redis.Redis")
    def test_update_expiry(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.from_url.return_value = mock_client
        storage = chat_storage.RedisConversationStorage()
        storage.redis_client = mock_client
        cid = "test-cid"
        mock_client.exists.return_value = True
        assert storage.update_expiry(cid)
        mock_client.exists.return_value = False
        assert not storage.update_expiry(cid)
