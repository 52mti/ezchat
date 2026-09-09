from __future__ import annotations

import json

from services.rule_learning import RuleLearningStore, anonymous_id


def test_candidates_are_app_scoped_and_metadata_is_anonymous(tmp_path):
    store = RuleLearningStore(tmp_path / "runtime" / "rule-learning")

    assert store.append_keywords(
        "xinyu",
        ["平台消息", "平台消息"],
        client_id="client-a",
        device_id="192.168.1.2",
    ) == 1

    path = tmp_path / "runtime" / "rule-learning" / "xinyu" / "rules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["rules"][0]
    assert item["keyword"] == "平台消息"
    pending_files = list((path.parent / "pending").glob("*.json"))
    assert len(pending_files) == 1
    event = json.loads(pending_files[0].read_text(encoding="utf-8"))
    assert event["client_id"] == anonymous_id("client-a", salt="xinyu")
    assert event["device_id"] == anonymous_id("192.168.1.2", salt="xinyu")
    assert "client-a" not in path.read_text(encoding="utf-8")
    assert "192.168.1.2" not in path.read_text(encoding="utf-8")
    assert not (tmp_path / "runtime" / "rule-learning" / "huanyou").exists()


def test_duplicate_keywords_are_not_added_again(tmp_path):
    store = RuleLearningStore(tmp_path / "rules")
    assert store.append_keywords("xinyu", ["平台消息"]) == 1
    assert store.append_keywords("xinyu", ["平台消息", "  平台消息  "]) == 0
    assert len(store.load_keywords("xinyu")) == 1


def test_remote_rules_are_merged_by_app(tmp_path, monkeypatch):
    store = RuleLearningStore(tmp_path / "rules")
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"app": "xinyu", "rules": [{"keyword": "共享平台消息"}]}

    monkeypatch.setattr("services.rule_learning.requests.get", lambda *args, **kwargs: Response())
    assert store.sync("xinyu", rules_url="https://example.invalid/{app}/rules.json") == ["共享平台消息"]
    assert store.load_keywords("xinyu") == ["共享平台消息"]
