from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeminiConfig:
    """
    Configuration for Gemini generation.
    """
    api_key: str
    model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 2048


class GeminiClient:
    """
    Gemini generation client with a small surface area.
    Prefer google.genai (new SDK). Keep the interface stable for swapping later.
    """

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config

        # Lazy import keeps package optional until you actually generate.
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Missing dependency: google-genai. Add it to dependencies and reinstall with uv."
            ) from e

        self._genai = genai
        self._client = genai.Client(api_key=config.api_key)

    # This is a function to generate text from Gemini using a single prompt string.
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        # google.genai supports “contents” as a string or structured parts.
        # We keep it simple and robust.
        contents: Any
        if system_prompt:
            contents = [
                {"role": "system", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": prompt}]},
            ]
        else:
            contents = prompt

        resp = self._client.models.generate_content(
            model=self._config.model,
            contents=contents,
            config={
                "temperature": self._config.temperature,
                "max_output_tokens": self._config.max_output_tokens,
            },
        )

        # This is a function to safely read text output across SDK response shapes.
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Fallback: try digging into candidates/parts if needed
        candidates = getattr(resp, "candidates", None)
        if candidates:
            try:
                parts = candidates[0].content.parts
                joined = "".join([p.text for p in parts if getattr(p, "text", None)])
                return joined.strip()
            except Exception:
                pass

        raise RuntimeError("Gemini response did not contain readable text output.")
