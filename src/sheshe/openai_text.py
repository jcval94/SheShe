"""Utilities to convert region interpretation cards into natural language using OpenAI."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import os
from contextlib import suppress

try:  # optional dependency
    from openai import OpenAI  # type: ignore
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore

try:  # pragma: no cover - importlib is stdlib but handle backports
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


class OpenAIRegionInterpreter:
    """Generate natural language descriptions for region cards with OpenAI.

    Parameters
    ----------
    model:
        Name of the chat completion model. Defaults to ``"gpt-4o-mini"``.
    api_key:
        Optional key used when constructing the OpenAI client. It is ignored
        when ``client`` is provided.
    language:
        Default language for the generated text. Can be overridden per call.
    temperature:
        Default sampling temperature. Can be overridden per call.
    client:
        Pre-instantiated OpenAI client. Mainly useful for testing/mocking.
    **completion_args:
        Additional arguments passed directly to ``chat.completions.create``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        *,
        language: str = "es",
        temperature: float = 0.0,
        client: Optional[Any] = None,
        **completion_args: Any,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            if OpenAI is None:  # pragma: no cover - runtime guard
                raise ImportError(
                    "openai package is required to use OpenAIRegionInterpreter"
                )
            _check_openai_version()
            api_key_resolved = self._resolve_api_key(api_key)
            if api_key_resolved is None:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY/OPENAI_KEY "
                    "or pass api_key explicitly."
                )
            self.client = OpenAI(api_key=api_key_resolved)
        self.model = model
        self.language = language
        self.temperature = temperature
        self.completion_args: Dict[str, Any] = {
            "max_tokens": 150,
        }
        self.completion_args.update(completion_args)

    # ------------------------------------------------------------------
    def _resolve_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Attempt to discover an API key from common locations."""
        if api_key:
            return api_key
        for var in ("OPENAI_API_KEY", "OPENAI_KEY"):
            val = os.getenv(var)
            if val:
                return val
        with suppress(Exception):  # Google Colab secret store
            from google.colab import userdata  # type: ignore
            val = userdata.get("OPENAI_API_KEY")
            if val:
                return val
        return None

    # ------------------------------------------------------------------
    def _card_to_prompt(self, card: Dict[str, Any], language: str) -> str:
        """Build a textual summary of a region card for prompting."""
        parts: List[str] = [
            f"ID={card.get('cluster_id')}",
            f"Etiqueta={card.get('label')}",
            f"Centro={card.get('center')}",
        ]
        box = "; ".join(card.get("box_rules", []))
        if box:
            parts.append(f"Caja: {box}")
        pairwise_parts = []
        for pr in card.get("pairwise_rules", []):
            pair = pr.get("pair", ("", ""))
            rules = ", ".join(pr.get("rules", []))
            pairwise_parts.append(f"{pair[0]} vs {pair[1]}: {rules}")
        if pairwise_parts:
            parts.append("Proyecciones: " + "; ".join(pairwise_parts))
        notes = "; ".join(card.get("notes", []))
        if notes:
            parts.append(f"Notas: {notes}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    def describe_cards(
        self,
        cards: Iterable[Dict[str, Any]],
        *,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        layout: Optional[str] = None,
    ) -> List[str]:
        """Return natural language descriptions for each region card.

        Parameters
        ----------
        language:
            Output language. Defaults to the interpreter's ``language`` value.
        temperature:
            Sampling temperature. Defaults to the interpreter's ``temperature``.
        layout:
            Optional textual description of the desired output format. When
            ``None`` the model responds freely.
        """

        lang = language or self.language
        temp = self.temperature if temperature is None else temperature
        call_args = {**self.completion_args, "temperature": temp}

        texts: List[str] = []
        for card in cards:
            region_summary = self._card_to_prompt(card, lang)
            system_msg = (
                f"Eres un asistente experto en análisis de datos. "
                f"Describe en {lang} la región de clúster con esta información."
            )
            user_msg = f"Información de la región: {region_summary}."
            if layout:
                user_msg += f"\nUsa este formato: {layout}"
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    **call_args,
                )
                texts.append(resp.choices[0].message.content.strip())
            except OpenAIError as exc:  # pragma: no cover - depends on API
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        return texts

__all__ = ["OpenAIRegionInterpreter"]


def _check_openai_version() -> None:
    """Ensure a recent ``openai`` package is installed."""
    try:
        version = importlib_metadata.version("openai")
    except Exception as exc:  # pragma: no cover - missing package
        raise ImportError("openai package is required") from exc
    major = int(version.split(".")[0])
    if major < 1:  # pragma: no cover - defensive guard
        raise ImportError(
            f"openai>=1.0 required, found {version}. Please upgrade the openai package."
        )
