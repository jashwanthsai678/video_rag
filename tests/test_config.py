"""Environment-variable coercion and config invariants."""

import pytest

from video_rag.config import Config, _env


class TestEnvCoercion:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("VIDEORAG_TEST_KEY", raising=False)
        assert _env("VIDEORAG_TEST_KEY", 7) == 7

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False), ("maybe", False),
    ])
    def test_bool_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("VIDEORAG_TEST_KEY", raw)
        assert _env("VIDEORAG_TEST_KEY", True) is expected

    def test_bool_is_checked_before_int(self, monkeypatch):
        """bool subclasses int, so the order of isinstance checks matters:
        a bool default must never be coerced through int()."""
        monkeypatch.setenv("VIDEORAG_TEST_KEY", "true")
        assert _env("VIDEORAG_TEST_KEY", False) is True

    def test_numeric_parsing(self, monkeypatch):
        monkeypatch.setenv("VIDEORAG_TEST_KEY", "12")
        assert _env("VIDEORAG_TEST_KEY", 0) == 12
        monkeypatch.setenv("VIDEORAG_TEST_KEY", "0.25")
        assert _env("VIDEORAG_TEST_KEY", 1.0) == pytest.approx(0.25)

    def test_strings_pass_through(self, monkeypatch):
        monkeypatch.setenv("VIDEORAG_TEST_KEY", "  spaced  ")
        assert _env("VIDEORAG_TEST_KEY", "default") == "  spaced  "


class TestConfig:
    def test_construction_creates_no_directories(self, tmp_path, monkeypatch):
        """Regression: Config used to mkdir in __post_init__, so merely
        importing and instantiating it littered the working directory."""
        monkeypatch.chdir(tmp_path)
        Config(data_dir=str(tmp_path / "d"), frames_dir=str(tmp_path / "f"))
        assert list(tmp_path.iterdir()) == []

    def test_ensure_dirs_is_explicit_and_idempotent(self, tmp_path):
        config = Config(
            data_dir=str(tmp_path / "data"),
            frames_dir=str(tmp_path / "data" / "frames"),
            clips_dir=str(tmp_path / "data" / "clips"),
        )
        config.ensure_dirs()
        config.ensure_dirs()
        assert (tmp_path / "data" / "frames").is_dir()
        assert (tmp_path / "data" / "clips").is_dir()

    def test_env_overrides_reach_the_dataclass(self, monkeypatch):
        monkeypatch.setenv("VIDEORAG_TOP_K", "3")
        monkeypatch.setenv("VIDEORAG_OCR_ENABLED", "false")
        config = Config()
        assert config.top_k == 3
        assert config.ocr_enabled is False

    def test_api_key_falls_back_to_provider_specific_names(self, monkeypatch):
        monkeypatch.delenv("VIDEORAG_LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert Config().llm_api_key == "sk-test"

    def test_ocr_languages_parse_to_a_list(self):
        assert Config(ocr_languages="en, hi ,te").ocr_language_list == ["en", "hi", "te"]

    def test_bge_query_instruction_is_present_by_default(self):
        """BGE is asymmetric: dropping the query prefix silently costs recall."""
        assert "searching relevant passages" in Config().query_instruction
