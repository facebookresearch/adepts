"""Shared multi-provider LLM client for ADEPTS benchmarks.

Supports OpenAI-compatible, Anthropic (Claude), Google (Gemini), and the
Hugging Face Inference Providers router (Qwen vision-language models).
"""

from __future__ import annotations

import base64
import os
from enum import Enum

from openai import AsyncOpenAI

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from google import genai
    from google.genai import types as gemini_types
except ImportError:
    genai = None
    gemini_types = None

# Qwen models are served through the Hugging Face Inference Providers router,
# which is OpenAI-compatible, so it reuses the AsyncOpenAI client. The router
# only auto-routes to providers enabled on your account, so we pin the provider
# explicitly per model (override for all Qwen models via the HF_PROVIDER env var,
# or clear it to fall back to the router's automatic selection).
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
HF_DEFAULT_PROVIDER = "featherless-ai"
HF_PROVIDERS = {
    "Qwen/Qwen3-VL-4B-Instruct": "featherless-ai",
    "Qwen/Qwen3-VL-8B-Instruct": "featherless-ai",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "novita",
}
# featherless serves Qwen3-VL-4B with a 32768-token total context, so the
# requested output must leave room for the prompt + image tokens. Bounding the
# output here keeps callers' large max_tokens defaults from overflowing that
# window; the tasks emit short actions, so this never truncates real output.
HF_MAX_OUTPUT_TOKENS = 8192


class ModelType(Enum):
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    GEMINI_2_5_CUA = "gemini-2.5-computer-use-preview-10-2025"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
    GPT_5_4 = "gpt-5.4"
    GPT_5_5 = "gpt-5.5-2026-04-23"
    QWEN_3_VL_235B_A22B = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    QWEN_3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
    QWEN_3_VL_4B = "Qwen/Qwen3-VL-4B-Instruct"


def resolve_models(names: list[str]) -> list[ModelType]:
    value_to_enum = {m.value: m for m in ModelType}
    models = []
    for name in names:
        if name not in value_to_enum:
            available = "\n".join(f"  {m.value}" for m in ModelType)
            raise SystemExit(f"Error: unknown model '{name}'. Available models:\n{available}")
        models.append(value_to_enum[name])
    return models


_openai_client: AsyncOpenAI | None = None
_anthropic_client = None
_gemini_client = None
_hf_client: AsyncOpenAI | None = None


def init_clients(
    openai_api_key: str | None = None,
    claude_api_key: str | None = None,
    gemini_api_key: str | None = None,
    hf_api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    global _openai_client, _anthropic_client, _gemini_client, _hf_client

    if openai_api_key:
        kwargs = {"api_key": openai_api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client = AsyncOpenAI(**kwargs)

    if claude_api_key:
        if AsyncAnthropic is None:
            raise ImportError("Install anthropic: pip install anthropic")
        _anthropic_client = AsyncAnthropic(api_key=claude_api_key)

    if gemini_api_key:
        if genai is None:
            raise ImportError("Install google-genai: pip install google-genai")
        _gemini_client = genai.Client(api_key=gemini_api_key)

    if hf_api_key:
        _hf_client = AsyncOpenAI(api_key=hf_api_key, base_url=HF_ROUTER_BASE_URL)


def is_claude_model(model: ModelType) -> bool:
    return model.value.startswith("claude-")


def is_gemini_model(model: ModelType) -> bool:
    return model.value.startswith("gemini-")


def is_qwen_model(model: ModelType) -> bool:
    return model.value.startswith("Qwen/")


def check_models_have_clients(models: list[ModelType]) -> None:
    missing = []
    for model in models:
        if is_claude_model(model) and _anthropic_client is None:
            missing.append((model.value, "CLAUDE_API_KEY or --claude-api-key"))
        elif is_gemini_model(model) and _gemini_client is None:
            missing.append((model.value, "GEMINI_API_KEY"))
        elif is_qwen_model(model) and _hf_client is None:
            missing.append((model.value, "HF_TOKEN"))
        elif not is_claude_model(model) and not is_gemini_model(model) and not is_qwen_model(model) and _openai_client is None:
            missing.append((model.value, "OPENAI_API_KEY or --api-key"))
    if missing:
        errors = "\n".join(f"  model '{name}' requires {hint}" for name, hint in missing)
        raise SystemExit(f"Error: missing API keys:\n{errors}")


def get_temperature(model: ModelType) -> float | None:
    if model == ModelType.CLAUDE_OPUS_4_7:
        return None
    return 0.0


async def chat_completion(
    model: ModelType, messages: list[dict], max_tokens: int = 1000,
    temperature_override: float | None = None,
) -> str:
    if is_claude_model(model):
        return await _claude_chat_completion(model, messages, max_tokens, temperature_override)
    if is_gemini_model(model):
        return await _gemini_chat_completion(model, messages, max_tokens, temperature_override)
    if is_qwen_model(model):
        return await _hf_chat_completion(model, messages, max_tokens, temperature_override)
    return await _openai_chat_completion(model, messages, max_tokens, temperature_override)


async def _openai_chat_completion(model, messages, max_tokens=1000, temperature_override=None):
    if _openai_client is None:
        raise RuntimeError(f"OpenAI client not initialized. Set OPENAI_API_KEY or pass --api-key to use {model.value}")
    temperature = temperature_override if temperature_override is not None else get_temperature(model)
    kwargs = {
        "model": model.value,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = await _openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.replace("```", "").strip()


async def _claude_chat_completion(model, messages, max_tokens=1000, temperature_override=None):
    if _anthropic_client is None:
        raise RuntimeError(f"Anthropic client not initialized. Set CLAUDE_API_KEY or pass --claude-api-key to use {model.value}")
    system_prompt = None
    claude_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_prompt = content if isinstance(content, str) else str(content)
            continue
        if isinstance(content, str):
            claude_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            claude_content = []
            for block in content:
                if block["type"] == "text":
                    claude_content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        media_type, _, b64_data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        })
            claude_messages.append({"role": role, "content": claude_content})
    collected = []
    stream_kwargs = {"model": model.value, "messages": claude_messages, "max_tokens": max_tokens}
    if system_prompt:
        stream_kwargs["system"] = system_prompt
    if model != ModelType.CLAUDE_OPUS_4_7:
        temp = temperature_override if temperature_override is not None else get_temperature(model)
        if temp is not None:
            stream_kwargs["temperature"] = temp
    async with _anthropic_client.messages.stream(**stream_kwargs) as stream:
        async for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).replace("```", "").strip()


async def _gemini_chat_completion(model, messages, max_tokens=1000, temperature_override=None):
    if _gemini_client is None:
        raise RuntimeError(f"Gemini client not initialized. Set GEMINI_API_KEY to use {model.value}")
    gemini_contents = []
    system_texts = []
    for msg in messages:
        content = msg["content"]
        # System prompts belong in Gemini's dedicated system_instruction field, not
        # as a conversation turn. Mapping them to a "model" turn makes the request
        # start with a model role, which the computer-use model rejects with a 400.
        if msg["role"] == "system":
            system_texts.append(content if isinstance(content, str) else str(content))
            continue
        role = "user" if msg["role"] == "user" else "model"
        if isinstance(content, str):
            gemini_contents.append(
                gemini_types.Content(role=role, parts=[gemini_types.Part(text=content)])
            )
        elif isinstance(content, list):
            parts = []
            for block in content:
                if block["type"] == "text":
                    parts.append(gemini_types.Part(text=block["text"]))
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        media_type, _, b64_data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        parts.append(gemini_types.Part(inline_data=gemini_types.Blob(
                            mime_type=media_type,
                            data=base64.b64decode(b64_data),
                        )))
            gemini_contents.append(gemini_types.Content(role=role, parts=parts))

    config_kwargs = {
        "max_output_tokens": max_tokens,
        "temperature": temperature_override if temperature_override is not None else 0.0,
    }
    if system_texts:
        config_kwargs["system_instruction"] = "\n".join(system_texts)
    if model == ModelType.GEMINI_2_5_CUA:
        config_kwargs["tools"] = [gemini_types.Tool(computer_use=gemini_types.ComputerUse())]

    response = await _gemini_client.aio.models.generate_content(
        model=model.value,
        contents=gemini_contents,
        config=gemini_types.GenerateContentConfig(**config_kwargs),
    )

    parts = response.candidates[0].content.parts if response.candidates else []
    text_parts = []
    for part in parts:
        if part.text:
            text_parts.append(part.text)
        elif part.function_call:
            fc = part.function_call
            args = fc.args or {}
            text_parts.append(
                f'<atem:function_calls><atem:invoke name="mobile.{fc.name}">'
                + "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>' for k, v in args.items())
                + "</atem:invoke></atem:function_calls>"
            )
    return "".join(text_parts).replace("```", "").strip()


async def _hf_chat_completion(model, messages, max_tokens=1000, temperature_override=None):
    if _hf_client is None:
        raise RuntimeError(f"Hugging Face client not initialized. Set HF_TOKEN to use {model.value}")
    temp = temperature_override if temperature_override is not None else 0.0
    # Pin the inference provider (e.g. "Qwen/...:featherless-ai") so routing does
    # not depend on which providers are enabled on the account. Set HF_PROVIDER=""
    # to fall back to the router's automatic provider selection.
    provider = os.environ.get("HF_PROVIDER")
    if provider is None:
        provider = HF_PROVIDERS.get(model.value, HF_DEFAULT_PROVIDER)
    model_id = f"{model.value}:{provider}" if provider else model.value
    response = await _hf_client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=min(max_tokens, HF_MAX_OUTPUT_TOKENS),
        temperature=temp,
    )
    return response.choices[0].message.content.replace("```", "").strip()
