"""
CLI Tool for Fake News Detection
Executes the AI Agent pipeline: ML Classifier + NewsAPI/RSS Search + Gemini LLM Synthesis
"""

import argparse
import json
import sys

try:
    from .agent import LLMSearchAgent
    from .ml_model import is_model_loaded
except ImportError:
    from agent import LLMSearchAgent
    from ml_model import is_model_loaded


def run_prompt(prompt: str, provider: str = None, page_size: int = 5):
    agent = LLMSearchAgent(provider=provider)
    return agent.analyze(prompt, page_size=page_size)


def main():
    p = argparse.ArgumentParser(description='FakeNewsAI: Dual-branch AI Agent Fact-Checker')
    p.add_argument('--prompt', '-p', help='Query or claim to analyze', required=False)
    p.add_argument('--provider', help='LLM provider override (gemini|hf)', default=None)
    p.add_argument('--page-size', type=int, default=5)
    args = p.parse_args()

    if not args.prompt:
        try:
            print("==================================================")
            print("  FakeNewsAI Agent: ML Classifier + Live News + LLM")
            print(f"  Model Loaded: {is_model_loaded()}")
            print("==================================================")
            args.prompt = input('Enter news claim or article to verify: ')
        except EOFError:
            print('No prompt provided, exiting')
            return

    if not args.prompt.strip():
        print("Empty prompt provided.")
        return

    print("\n[AI Agent] Running dual-branch analysis...")
    res = run_prompt(args.prompt, provider=args.provider, page_size=args.page_size)

    final = res.get('final_analysis', {})
    ml = res.get('ml_classifier', {})
    sources = res.get('news_sources', [])

    print("\n" + "=" * 60)
    print(f"  VERDICT: {final.get('verdict', 'Unknown').upper()} (Confidence: {int(final.get('confidence_score', 0)*100)}%)")
    print("=" * 60)
    print(f"\n[Executive Summary]\n{final.get('executive_summary', 'N/A')}")
    
    print(f"\n[ML Statistical Classifier (44.9k articles)]")
    print(f"  • Predicted Label:  {ml.get('label', 'unknown').upper()}")
    print(f"  • Fake Probability: {ml.get('fake_probability', 0)*100:.1f}%")
    print(f"  • Real Probability: {ml.get('real_probability', 0)*100:.1f}%")
    signals = ml.get('top_signals', [])
    if signals:
        print(f"  • Key Signals:      {', '.join([s['word'] for s in signals])}")

    print(f"\n[Live News Evidence ({len(sources)} sources found)]")
    for idx, s in enumerate(sources[:3], 1):
        print(f"  {idx}. [{s.get('source')}] {s.get('title')}")
        if s.get('url'):
            print(f"     URL: {s.get('url')}")

    red_flags = final.get('red_flags', [])
    if red_flags:
        print("\n[Red Flags / Risk Factors]")
        for rf in red_flags:
            print(f"  [!] {rf}")

    rec = final.get('recommendations')
    if rec:
        print(f"\n[Recommendation]\n  [*] {rec}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
