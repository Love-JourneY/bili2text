from pathlib import Path

from b2t.config import Settings
from b2t.user_config import AppConfig


def test_app_config_round_trip(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    config = AppConfig(
        default_provider="qwen3",
        default_model="C:/models/qwen3-asr-0.6b-int8",
        language="en-US",
    )
    config.enabled_features = ["web", "window"]
    config.qwen3.model_dir = "C:/models/qwen3-asr-0.6b-int8"
    config.qwen3.vad_model = "C:/models/silero_vad.onnx"
    config.qwen3.num_threads = 6
    config.qwen3.provider = "cuda"
    config.volcengine.api_key = "secret"
    config.save(settings)

    loaded = AppConfig.load(settings)
    assert loaded.language == "en-US"
    assert loaded.enabled_features == ["web", "window"]
    assert loaded.default_provider == "qwen3"
    assert loaded.default_model == "C:/models/qwen3-asr-0.6b-int8"
    assert loaded.qwen3.model_dir == "C:/models/qwen3-asr-0.6b-int8"
    assert loaded.qwen3.vad_model == "C:/models/silero_vad.onnx"
    assert loaded.qwen3.num_threads == 6
    assert loaded.qwen3.provider == "cuda"
    assert loaded.volcengine.api_key == "secret"


def test_legacy_whisper_config_is_migrated_to_qwen3(tmp_path: Path) -> None:
    """老配置(写着 whisper/sensevoice)必须能被静默迁移,不能把程序卡死。"""
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    settings.config_path.write_text(
        '{"language": "zh-CN", "enabled_providers": ["whisper", "sensevoice"],'
        ' "default_provider": "whisper", "default_model": "medium"}',
        encoding="utf-8",
    )

    loaded = AppConfig.load(settings)
    assert loaded.default_provider == "qwen3"
    assert loaded.enabled_providers == ["qwen3", "qwen3"]
    # whisper 的 small/medium/large 对新引擎没有意义,应被清空
    assert loaded.default_model == ""

