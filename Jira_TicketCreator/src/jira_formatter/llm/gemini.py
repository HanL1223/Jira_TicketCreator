from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiConfig:
    """
    Configuration for Gemini generation.
    """
    api_key: str
    model: str = "gemini-2.5-flash-lite-preview-09-2025"
    temperature: float = 0.2
    max_output_tokens: int = 2048


class GeminiClient:
    """
    Gemini generation client with a small surface area.
    Uses google.genai (new SDK). Keep the interface stable for swapping later.
    """

    def __init__(self, config: GeminiConfig) -> None:
        # This is a function to initialise a Gemini client using the provided API key + model settings.
        self._config = config

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Missing dependency: google-genai. Add it to dependencies and reinstall with uv."
            ) from e

        self._types = types
        self._client = genai.Client(api_key=config.api_key)

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """
        This is a function to generate text from Gemini using a user prompt, optionally with a system instruction.

        Notes:
        - Gemini API Content roles must be 'user' or 'model' (no 'system' role).
        - System behavior should be provided via GenerateContentConfig(system_instruction=...).
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        cfg = self._types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            system_instruction=system_prompt or None,
        )

        resp = self._client.models.generate_content(
            model=self._config.model,
            contents=prompt,  # simple + robust: treat prompt as the user message
            config=cfg,
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
                if joined.strip():
                    return joined.strip()
            except Exception:
                pass

        raise RuntimeError("Gemini response did not contain readable text output.")
