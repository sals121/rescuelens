import pytest

from rescuelens.config import Settings


def test_model_and_budget_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    monkeypatch.setenv("RESCUELENS_MAX_COST_USD", "2.5")
    settings = Settings(_env_file=None)
    assert settings.anthropic_model == "claude-test-model"
    assert settings.llm_max_cost_usd == 2.5


def test_api_key_is_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    settings = Settings(_env_file=None)
    assert "sk-ant-secret" not in repr(settings)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-secret"
