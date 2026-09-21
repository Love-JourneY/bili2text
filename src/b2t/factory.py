from __future__ import annotations

from pathlib import Path

from b2t.config import Settings
from b2t.downloaders import YtDlpDownloader
from b2t.pipeline import B2TPipeline
from b2t.transcribers import Qwen3LocalTranscriber
from b2t.user_config import AppConfig

# 本 fork 的本地引擎只有 Qwen3-ASR(ONNX)。
# 原版的 whisper(PyTorch+CUDA,venv 4.8GB)与 sensevoice(funasr-onnx+torch)
# 已按"运行时不要那么繁重"的要求整条移除。
LOCAL_PROVIDER_ALIASES = {"qwen3", "qwen3-asr", "qwen3_asr", "sherpa", "onnx"}


def build_pipeline(
    *,
    settings: Settings,
    config: AppConfig,
    provider: str | None = None,
    model: str | None = None,
) -> B2TPipeline:
    selected_provider = (provider or config.default_provider).strip().lower()
    selected_model = (model or config.default_model).strip()

    if selected_provider in LOCAL_PROVIDER_ALIASES:
        model_dir_text = selected_model or config.qwen3.model_dir
        if not model_dir_text:
            raise RuntimeError(
                "Qwen3-ASR 需要一个本地模型目录。先跑 `bili2text bootstrap`,"
                "或设置 config.json 里的 qwen3.model_dir。"
            )
        transcriber = Qwen3LocalTranscriber(
            model_dir=Path(model_dir_text).expanduser(),
            vad_model=Path(config.qwen3.vad_model).expanduser() if config.qwen3.vad_model else None,
            num_threads=config.qwen3.num_threads,
            provider=config.qwen3.provider,
            max_new_tokens=config.qwen3.max_new_tokens,
            max_total_len=config.qwen3.max_total_len,
        )
    elif selected_provider == "volcengine":
        from b2t.transcribers.volcengine import VolcengineFlashTranscriber

        transcriber = VolcengineFlashTranscriber(
            api_key=config.volcengine.api_key,
            app_key=config.volcengine.app_key,
            access_key=config.volcengine.access_key,
            resource_id=config.volcengine.resource_id,
            model_name=selected_model or config.volcengine.model_name,
            use_itn=config.volcengine.use_itn,
        )
    else:
        raise RuntimeError(
            f"不支持的 provider: {selected_provider}"
            "(本 fork 的本地引擎是 qwen3;云端可用 volcengine)"
        )

    return B2TPipeline(
        settings=settings,
        downloader=YtDlpDownloader(),
        transcriber=transcriber,
    )
