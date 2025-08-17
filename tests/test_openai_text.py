import pytest

from sheshe.openai_text import OpenAIRegionInterpreter


class DummyClient:
    def __init__(self) -> None:
        class _Completions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):  # noqa: D401 - simple mock
                self.calls.append(kwargs)
                msg = type("M", (), {"content": "texto"})()
                choice = type("C", (), {"message": msg})()
                return type("R", (), {"choices": [choice]})()

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


def test_describe_cards_uses_client():
    client = DummyClient()
    cards = [
        {
            "cluster_id": 0,
            "label": "A",
            "center": [0, 0],
            "box_rules": ["x <= 1"],
            "pairwise_rules": [],
            "notes": [],
        }
    ]
    expl = OpenAIRegionInterpreter(client=client, model="mock")
    texts = expl.describe_cards(cards, language="es", layout="lista", temperature=0.3)
    assert texts == ["texto"]
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "mock"
    assert call["temperature"] == 0.3
    assert "lista" in call["messages"][1]["content"]
    assert call["messages"][0]["role"] == "system"


class FailingClient:
    def __init__(self) -> None:
        class _Completions:
            def create(self, **kwargs):
                raise RuntimeError("boom")

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


def test_describe_cards_raises_runtime_error():
    client = FailingClient()
    cards = [
        {
            "cluster_id": 0,
            "label": "A",
            "center": [0, 0],
            "box_rules": [],
            "pairwise_rules": [],
            "notes": [],
        }
    ]
    expl = OpenAIRegionInterpreter(client=client, model="mock")
    with pytest.raises(RuntimeError):
        expl.describe_cards(cards)
