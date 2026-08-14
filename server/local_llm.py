import os
from typing import Any, Dict

_PIPE = None

def _init_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError('transformers is not installed. Run `pip install -r requirements.txt`.') from e

    model = os.getenv('LOCAL_HF_MODEL', 'gpt2')
    # Create a text-generation pipeline. Users may replace LOCAL_HF_MODEL with a larger model.
    _PIPE = pipeline('text-generation', model=model)
    return _PIPE


def generate(prompt: str, max_output_tokens: int = 64, temperature: float = 0.0) -> Dict[str, Any]:
    """Generate text using a local Hugging Face model.

    Returns a dict compatible with the agent's expected shape: {'candidates': [{'content': '...'}]}
    """
    pipe = _init_pipeline()
    # Many models use `max_new_tokens` for generation
    out = pipe(prompt, max_new_tokens=max_output_tokens, do_sample=False, temperature=temperature)
    # pipeline returns a list of dicts with 'generated_text' or 'text'
    text = ''
    if isinstance(out, list) and out:
        cand = out[0]
        text = cand.get('generated_text') or cand.get('text') or str(cand)
    else:
        text = str(out)

    return {'candidates': [{'content': text}]}
