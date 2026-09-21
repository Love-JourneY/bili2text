"""Qwen3-ASR(ONNX)引擎的单元测试。

只测**不依赖模型文件**的部分:输出解析、命令拼装、错误信息。
真机转写需要模型,放在端到端验收里做。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from b2t.transcribers.qwen3_local import (
    Qwen3LocalTranscriber,
    build_sherpa_missing_message,
    join_segments,
    parse_segments,
)


# -- 输出解析 ---------------------------------------------------------------
def test_parse_vad_style_segments() -> None:
    raw = "293.054 -- 313.340: 第一段内容\n\n313.822 -- 334.556: 第二段内容\n"
    segments = parse_segments(raw)
    assert len(segments) == 2
    assert segments[0]["start"] == pytest.approx(293.054)
    assert segments[0]["end"] == pytest.approx(313.340)
    assert segments[0]["text"] == "第一段内容"
    assert segments[1]["text"] == "第二段内容"


def test_parse_single_shot_json_segments() -> None:
    raw = '{"lang": "", "text": "整段一次的旧格式", "timestamps": []}'
    segments = parse_segments(raw)
    assert len(segments) == 1
    assert segments[0]["text"] == "整段一次的旧格式"
    assert segments[0]["start"] is None


def test_parse_ignores_noise_and_empty() -> None:
    assert parse_segments("") == []
    assert parse_segments(None) == []
    assert parse_segments("num threads: 8\ndecoding method: greedy_search\n") == []
    # 有时间戳但文本为空的段应被丢弃
    assert parse_segments("1.000 -- 2.000:   \n") == []


def test_join_segments_one_line_each() -> None:
    segments = [{"text": "甲"}, {"text": "乙"}, {"text": "丙"}]
    assert join_segments(segments) == "甲\n乙\n丙"


# -- 命令拼装与前置校验 -----------------------------------------------------
def _make_model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "qwen3-asr-0.6b-int8"
    (model / "tokenizer").mkdir(parents=True)
    (model / "conv_frontend.onnx").write_bytes(b"x")
    (model / "encoder.int8.onnx").write_bytes(b"x")
    (model / "decoder.int8.onnx").write_bytes(b"x")
    return model


def test_build_command_prefers_int8_and_passes_vad(tmp_path: Path) -> None:
    model = _make_model_dir(tmp_path)
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"x")
    transcriber = Qwen3LocalTranscriber(
        model_dir=model, vad_model=vad, num_threads=8, provider="cuda"
    )

    command = transcriber._build_command(
        "/usr/bin/sherpa-onnx-vad-with-offline-asr", model, Path("/tmp/a.wav")
    )

    assert command[0] == "/usr/bin/sherpa-onnx-vad-with-offline-asr"
    assert f"--silero-vad-model={vad}" in command
    assert f"--qwen3-asr-encoder={model / 'encoder.int8.onnx'}" in command
    assert f"--qwen3-asr-decoder={model / 'decoder.int8.onnx'}" in command
    assert "--num-threads=8" in command
    assert "--provider=cuda" in command
    assert command[-1] == "/tmp/a.wav"


def test_plain_offline_binary_gets_no_vad_flag(tmp_path: Path) -> None:
    """退化到 sherpa-onnx-offline(非 VAD 版)时不该硬塞 silero 参数。"""
    model = _make_model_dir(tmp_path)
    transcriber = Qwen3LocalTranscriber(model_dir=model, vad_model=tmp_path / "v.onnx")

    command = transcriber._build_command(
        "/usr/bin/sherpa-onnx-offline", model, Path("/tmp/a.wav")
    )

    assert not any(arg.startswith("--silero-vad-model") for arg in command)


def test_missing_model_dir_raises_clear_error(tmp_path: Path) -> None:
    transcriber = Qwen3LocalTranscriber(model_dir=tmp_path / "nope")
    with pytest.raises(RuntimeError, match="模型目录不存在"):
        transcriber._validate_model_dir()


def test_wrong_model_dir_raises_clear_error(tmp_path: Path) -> None:
    bogus = tmp_path / "sherpa-onnx-whisper-base"
    bogus.mkdir()
    transcriber = Qwen3LocalTranscriber(model_dir=bogus)
    with pytest.raises(RuntimeError, match="qwen3_asr"):
        transcriber._validate_model_dir()


def test_sherpa_missing_message_mentions_install() -> None:
    message = build_sherpa_missing_message()
    assert "sherpa-onnx" in message
    assert "pacman" in message
    assert "silero_vad.onnx" in message
