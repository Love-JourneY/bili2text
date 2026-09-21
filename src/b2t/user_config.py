from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from b2t.config import Settings
from b2t.i18n import DEFAULT_LANGUAGE, normalize_language

# 本 fork 只保留两个 provider:
#   qwen3      —— 本地,ONNX Runtime(sherpa-onnx),默认
#   volcengine —— 云端 API(可选)
# 原版的 whisper / sensevoice 已移除,理由见 transcribers/qwen3_local.py 顶部注释。
ALL_PROVIDERS = ("qwen3", "volcengine")
ALL_FEATURES = ("web", "server", "window")


@dataclass(slots=True)
class Qwen3Config:
    """Qwen3-ASR(ONNX)本地引擎配置。

    model_dir 指向 sherpa-onnx 的 qwen3_asr 模型目录,需含:
        conv_frontend.onnx / encoder*.onnx / decoder*.onnx / tokenizer/
    vad_model 指向 Silero VAD 的 onnx(长音频切块用,约 2MB)。
    provider 传给 onnxruntime:cpu / cuda / coreml(需对应 build 才生效)。
    """

    model_dir: str = ""
    vad_model: str = ""
    num_threads: int = 4
    provider: str = "cpu"
    max_new_tokens: int = 512
    max_total_len: int = 2048


@dataclass(slots=True)
class VolcengineConfig:
    api_key: str = ""
    app_key: str = ""
    access_key: str = ""
    resource_id: str = "volc.bigasr.auc_turbo"
    model_name: str = "bigmodel"
    use_itn: bool = True


def _qwen3_from(data: dict) -> Qwen3Config:
    """兼容旧配置:老版本这里放的是 sensevoice / whisper 的字段。"""
    raw = data.get("qwen3") or {}
    return Qwen3Config(**{k: v for k, v in raw.items() if k in Qwen3Config.__slots__})


@dataclass(slots=True)
class AppConfig:
    language: str = DEFAULT_LANGUAGE
    enabled_providers: list[str] = field(default_factory=lambda: ["qwen3"])
    enabled_features: list[str] = field(default_factory=lambda: ["window"])
    default_provider: str = "qwen3"
    # 对本地 Qwen3-ASR 而言,"model" 就是模型目录
    default_model: str = ""
    qwen3: Qwen3Config = field(default_factory=Qwen3Config)
    volcengine: VolcengineConfig = field(default_factory=VolcengineConfig)

    @classmethod
    def load(cls, settings: Settings) -> "AppConfig":
        if not settings.config_path.exists():
            return cls()

        data = json.loads(settings.config_path.read_text(encoding="utf-8"))
        enabled = data.get("enabled_providers")
        if enabled is None:
            # backwards compat: old configs only had default_provider
            enabled = [data.get("default_provider", "qwen3")]
        # 老配置里可能还写着 whisper/sensevoice —— 静默映射到 qwen3,别让老配置直接把程序卡死
        enabled = [
            "qwen3" if item in ("whisper", "sensevoice") else item for item in enabled
        ]
        default_provider = data.get("default_provider", "qwen3")
        if default_provider in ("whisper", "sensevoice"):
            default_provider = "qwen3"
        default_model = data.get("default_model", "")
        # 老配置的 default_model 是 whisper 的 small/medium/large,对新引擎无意义
        if default_model in ("tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"):
            default_model = ""
        features = data.get("enabled_features", ["window"])
        return cls(
            language=normalize_language(data.get("language")),
            enabled_providers=enabled,
            enabled_features=features,
            default_provider=default_provider,
            default_model=default_model,
            qwen3=_qwen3_from(data),
            volcengine=VolcengineConfig(**data.get("volcengine", {})),
        )

    def save(self, settings: Settings) -> None:
        settings.ensure_directories()
        settings.config_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
