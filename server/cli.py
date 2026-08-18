"""Command-line interface for the FakeNewsAI verification pipeline.

Usage::

    python -m server.cli --prompt "NASA confirms water on the Moon"
"""

import argparse
import json
import sys

from .agent import LLMSearchAgent
from .config import settings
from .logging_config import setup_logging
from .ml_model import get_model_accuracy_display, is_model_loaded


def run_prompt(prompt: str, provider: str = None, page_size: int = 5) -> dict:
    """Analyse ``prompt`` and return the full report dict."""
    agent = LLMSearchAgent(provider=provider)
    return agent.analyze(prompt, page_size=page_size)


def _clamp_page_size(value: int) -> int:
    """Clamp ``value`` into the configured page-size bounds."""
    return max(settings.MIN_PAGE_SIZE, min(settings.MAX_PAGE_SIZE, value))


def _print_report(res: dict) -> None:
    """Render a report dict as a human-readable terminal summary."""
    final = res.get("final_analysis", {})
    ml = res.get("ml_classifier", {})
    sources = res.get("news_sources", [])

    confidence = int(final.get("confidence_score", 0) * 100)
    print("\n" + "=" * 60)
    print(f"  VERDICT: {final.get('verdict', 'Unknown').upper()} "
          f"(Confidence: {confidence}%)")
    print("=" * 60)
    print(f"\n[Executive Summary]\n{final.get('executive_summary', 'N/A')}")

    print(f"\n[ML Classifier - accuracy {ml.get('model_accuracy', 'unknown')}]")
    print(f"  - Predicted Label:  {ml.get('label', 'unknown').upper()}")
    print(f"  - Fake Probability: {ml.get('fake_probability', 0) * 100:.1f}%")
    print(f"  - Real Probability: {ml.get('real_probability', 0) * 100:.1f}%")
    signals = ml.get("top_signals", [])
    if signals:
        print(f"  - Key Signals:      {', '.join(s['word'] for s in signals)}")

    print(f"\n[Live News Evidence ({len(sources)} source(s) found)]")
    for idx, src in enumerate(sources[:3], 1):
        print(f"  {idx}. [{src.get('source')}] {src.get('title')}")
        if src.get("url"):
            print(f"     URL: {src.get('url')}")

    red_flags = final.get("red_flags", [])
    if red_flags:
        print("\n[Red Flags / Risk Factors]")
        for flag in red_flags:
            print(f"  [!] {flag}")

    if final.get("recommendations"):
        print(f"\n[Recommendation]\n  [*] {final['recommendations']}")
    print("\n" + "=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FakeNewsAI: dual-branch AI agent fact-checker"
    )
    parser.add_argument("--prompt", "-p", help="Claim or article to analyse")
    parser.add_argument("--provider", default=None,
                        help="LLM provider override (hf|local)")
    parser.add_argument("--page-size", type=int,
                        default=settings.DEFAULT_PAGE_SIZE)
    parser.add_argument("--json", action="store_true",
                        help="Emit the raw JSON report instead of a summary")
    args = parser.parse_args()

    setup_logging(settings.LOG_LEVEL)

    if not args.prompt:
        print("=" * 55)
        print("  FakeNewsAI Agent: ML Classifier + Live News")
        print(f"  Model loaded: {is_model_loaded()} "
              f"(accuracy: {get_model_accuracy_display()})")
        print("=" * 55)
        try:
            args.prompt = input("Enter news claim or article to verify: ")
        except EOFError:
            print("No prompt provided; exiting.")
            return

    if not args.prompt.strip():
        print("Empty prompt provided.")
        return

    # Same cap the HTTP API enforces, so both entry points behave identically.
    if len(args.prompt) > settings.MAX_INPUT_CHARS:
        print(f"Input too long: {len(args.prompt)} chars "
              f"(max {settings.MAX_INPUT_CHARS}).", file=sys.stderr)
        raise SystemExit(2)

    res = run_prompt(args.prompt, provider=args.provider,
                     page_size=_clamp_page_size(args.page_size))

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_report(res)


if __name__ == "__main__":
    main()
