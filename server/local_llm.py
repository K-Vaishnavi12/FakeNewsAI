"""Optional local LLM backend (Hugging Face ``transformers``).

This path is **disabled by default** (``ENABLE_LLM=false``).

The configured Hugging Face model (defaulting to ``gpt2`` via ``settings.LOCAL_HF_MODEL``)
is loaded lazily on demand when LLM synthesis is explicitly enabled.
"""

import threading
from typing import Any, Dict, Optional

from .config import settings
from .logging_config import get_logger

logger = get_logger(__name__)

_PIPE = None
_TOKENIZER = None
_INIT_LOCK = threading.Lock()  # pipeline init is not thread-safe


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM path is disabled or the model cannot be loaded."""


def is_enabled() -> bool:
    """Return True if the operator explicitly opted in to LLM synthesis."""
    return bool(settings.ENABLE_LLM)


def _init_pipeline():
    """Lazily build (and memoise) the text-generation pipeline.

    Raises:
        LLMUnavailableError: If disabled, if ``transformers`` is missing, or
            if the model fails to load.
    """
    global _PIPE, _TOKENIZER

    if not is_enabled():
        raise LLMUnavailableError(
            "LLM synthesis is disabled. Set ENABLE_LLM=true to enable it."
        )

    if _PIPE is not None:
        return _PIPE

    with _INIT_LOCK:
        if _PIPE is not None:  # another thread won the race
            return _PIPE

        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise LLMUnavailableError(
                "transformers is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        model_name = settings.LOCAL_HF_MODEL
        token = settings.HF_TOKEN or None
        logger.info("Loading LLM '%s' (this may take a while)...", model_name)

        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name, token=token)
            _PIPE = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=_TOKENIZER,
                token=token,
            )
        except Exception as exc:
            # Model loading can fail in many library-specific ways (network,
            # OOM, missing weights); surface it rather than swallowing it.
            logger.error("Failed to load LLM '%s'", model_name, exc_info=True)
            raise LLMUnavailableError(
                f"Could not load model '{model_name}': {exc}"
            ) from exc

        logger.info("LLM '%s' ready.", model_name)
        return _PIPE


def _build_inputs(prompt: str, system_prompt: Optional[str]) -> Any:
    """Apply the model's chat template if it has one, else return raw text."""
    if _TOKENIZER is not None and getattr(_TOKENIZER, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


def generate(prompt: str, max_output_tokens: int = 512,
             temperature: float = 0.2,
             system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Generate a completion for ``prompt``.

    Args:
        prompt: The user-role content.
        max_output_tokens: Generation budget.
        temperature: Sampling temperature; ``<= 0`` switches to greedy decoding.
        system_prompt: Optional system-role instructions.

    Returns:
        ``{'candidates': [{'content': str}]}`` -- the shape ``agent.py`` expects.

    Raises:
        LLMUnavailableError: If the LLM path is disabled or unloadable.
    """
    pipe = _init_pipeline()
    model_input = _build_inputs(prompt, system_prompt)

    # `temperature` is only a valid argument when sampling is on; passing both
    # do_sample=False and a temperature emitted a warning and was ignored.
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_output_tokens,
        "return_full_text": False,  # we want only the completion
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        gen_kwargs["do_sample"] = False

    out = pipe(model_input, **gen_kwargs)

    text = ""
    if isinstance(out, list) and out:
        candidate = out[0]
        if isinstance(candidate, dict):
            text = candidate.get("generated_text") or candidate.get("text") or ""
        else:
            text = str(candidate)

    return {"candidates": [{"content": text}]}


def reset() -> None:
    """Drop the loaded pipeline (frees memory; mainly useful in tests)."""
    global _PIPE, _TOKENIZER
    with _INIT_LOCK:
        _PIPE = None
        _TOKENIZER = None
