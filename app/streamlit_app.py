"""Step 10 — VeriTruth Streamlit frontend.

Talks only to the FastAPI backend over HTTP, so it stays deployable as its own
container. Every network call is wrapped and rendered as a friendly message
rather than a traceback.

Run::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("VERITRUTH_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("VERITRUTH_UI_TIMEOUT", "90"))

BAND_STYLE: dict[str, dict[str, str]] = {
    "Real": {
        "color": "#0f7b3f",
        "bg": "#ecfdf3",
        "edge": "#a6f0c6",
        "icon": "OK",
        "label": "LIKELY REAL",
        "blurb": "Consistent with verified reporting.",
    },
    "Suspicious": {
        "color": "#b25e02",
        "bg": "#fffaeb",
        "edge": "#fedf89",
        "icon": "!",
        "label": "SUSPICIOUS",
        "blurb": "Mixed signals — verify before sharing.",
    },
    "Fake": {
        "color": "#b42318",
        "bg": "#fef3f2",
        "edge": "#fecdc9",
        "icon": "X",
        "label": "LIKELY FAKE",
        "blurb": "Strong markers of fabricated content.",
    },
}

STATUS_ICON = {"supported": "SUPPORTED", "refuted": "REFUTED", "unverified": "UNVERIFIED"}

SAMPLES: dict[str, str] = {
    "Real": (
        "The Reserve Bank of India kept its benchmark repo rate unchanged at 6.5 percent "
        "on Friday, citing steady inflation and resilient domestic demand. The monetary "
        "policy committee voted four to two in favour of the decision, and the governor "
        "said the central bank would continue to monitor food prices closely."
    ),
    "Fake": (
        "BREAKING!!! Doctors are SHOCKED: a secret kitchen ingredient the government has "
        "banned for 40 years reverses ageing overnight and cures every known disease. "
        "Big Pharma is desperately trying to delete this video before you see the truth. "
        "SHARE before it is removed forever!!!"
    ),
    "Borderline": (
        "A new study circulating on social media claims that remote workers are twice as "
        "productive as office workers, but researchers have not yet released the "
        "underlying data and the sample size appears to be small."
    ),
}


st.set_page_config(
    page_title="VeriTruth",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
  /* Force a light surface even if the viewer's OS is in dark mode. */
  .stApp, [data-testid="stAppViewContainer"] { background: #f7f9fc; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6eaf2;
  }

  /* Hero banner */
  .vt-hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #a855f7 100%);
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 22px;
    color: #ffffff;
    box-shadow: 0 12px 30px -14px rgba(79,70,229,.65);
  }
  .vt-hero h1 {
    margin: 0; font-size: 34px; font-weight: 800; letter-spacing: -.02em;
  }
  .vt-hero p { margin: 6px 0 0; font-size: 15px; opacity: .92; }

  /* Generic white card */
  .vt-card {
    background: #ffffff;
    border: 1px solid #e6eaf2;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px -4px rgba(16,24,40,.12);
  }

  /* Section headings */
  .vt-section {
    font-size: 12px; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: #667085;
    margin: 22px 0 8px;
  }

  /* Verdict card */
  .vt-verdict { border-radius: 16px; padding: 22px 26px; margin-bottom: 6px; }
  .vt-verdict .vt-eyebrow {
    font-size: 12px; letter-spacing: .16em; font-weight: 700; opacity: .85;
  }
  .vt-verdict .vt-label {
    font-size: 38px; font-weight: 800; line-height: 1.15; margin-top: 2px;
  }
  .vt-verdict .vt-blurb { font-size: 15px; color: #475467; margin-top: 6px; }
  .vt-pill {
    display: inline-block; font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 999px; margin-left: 8px;
    background: rgba(255,255,255,.75); border: 1px solid rgba(16,24,40,.12);
    color: #475467;
  }

  /* Article body with token highlights */
  .vt-article {
    background: #ffffff; border: 1px solid #e6eaf2; border-radius: 14px;
    padding: 18px 22px; line-height: 2.05; font-size: 16px; color: #101828;
  }
  .vt-tok { border-radius: 5px; padding: 2px 4px; cursor: help; }

  /* Buttons */
  .stButton > button {
    border-radius: 10px; font-weight: 600; border: 1px solid #d9dfea;
    transition: transform .06s ease, box-shadow .16s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px -8px rgba(16,24,40,.45);
  }

  /* Inputs and expanders */
  .stTextArea textarea {
    border-radius: 12px; border: 1px solid #d9dfea; font-size: 15px;
    background: #ffffff; color: #101828;
  }
  [data-testid="stExpander"] {
    border: 1px solid #e6eaf2; border-radius: 12px; background: #ffffff;
  }
  [data-testid="stProgress"] > div > div > div { border-radius: 999px; }
</style>
"""


# ------------------------------------------------------------------ API layer
def call_api(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """POST to the backend. Returns ``(json, error_message)`` — never raises."""
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot reach the VeriTruth API at {API_URL}. "
            "Start it with: uvicorn src.api.main:app --port 8000"
        )
    except requests.exceptions.Timeout:
        return None, f"The API did not respond within {REQUEST_TIMEOUT:.0f}s. Try shorter text."
    except Exception as exc:
        return None, f"Unexpected error contacting the API: {exc}"

    if response.status_code == 422:
        return None, "The text was rejected as invalid. Please enter at least one character."
    if response.status_code >= 400:
        return None, f"The API returned HTTP {response.status_code}. Check the backend logs."
    try:
        return response.json(), ""
    except ValueError:
        return None, "The API returned a malformed response."


def get_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


# ------------------------------------------------------------------ rendering
def render_verdict(verdict: str, score: float, degraded: bool) -> None:
    style = BAND_STYLE.get(verdict, BAND_STYLE["Suspicious"])
    pill = '<span class="vt-pill">degraded mode</span>' if degraded else ""
    st.markdown(
        f"""
        <div class="vt-verdict" style="background:{style['bg']};
             border:1px solid {style['edge']};
             box-shadow:0 10px 26px -16px {style['color']};">
          <div class="vt-eyebrow" style="color:{style['color']};">VERDICT{pill}</div>
          <div class="vt-label" style="color:{style['color']};">
            {style['icon']} &nbsp;{style['label']}
          </div>
          <div class="vt-blurb">{style['blurb']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 1])
    with left:
        st.progress(min(max(score / 100.0, 0.0), 1.0))
        st.caption("0-40 Fake  ·  40-70 Suspicious  ·  70-100 Real")
    with right:
        st.metric("Trust score", f"{score:.0f} / 100")


def highlight_text(text: str, tokens: list[dict[str, Any]]) -> str:
    """Wrap influential tokens in coloured spans (red = pushes toward Fake)."""
    weights = {
        str(t.get("word", "")).lower(): float(t.get("weight", 0.0))
        for t in tokens
        if str(t.get("word", "")).strip()
    }
    if not weights:
        return f"<div class='vt-article'>{html.escape(text)}</div>"
    peak = max(abs(w) for w in weights.values()) or 1.0

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        weight = weights.get(word.lower())
        escaped = html.escape(word)
        if weight is None or weight == 0:
            return escaped
        alpha = min(0.16 + 0.60 * abs(weight) / peak, 0.82)
        colour = f"rgba(240,68,56,{alpha:.2f})" if weight < 0 else f"rgba(18,183,106,{alpha:.2f})"
        direction = "toward Fake" if weight < 0 else "toward Real"
        return (
            f"<span class='vt-tok' title='{weight:+.4f} ({direction})' "
            f"style='background:{colour};'>{escaped}</span>"
        )

    body = re.sub(r"\b[\w'-]+\b", repl, html.escape(text))
    return f"<div class='vt-article'>{body}</div>"


def render_claims(claims: list[dict[str, Any]], citations: list[dict[str, Any]]) -> None:
    by_id = {int(c.get("id", 0)): c for c in citations}
    if not claims:
        st.info("The agent did not extract any independently checkable claims.")
        return
    for i, claim in enumerate(claims, 1):
        status = str(claim.get("status", "unverified"))
        label = STATUS_ICON.get(status, status.upper())
        with st.expander(f"Claim {i} — {label} — {claim.get('claim','')[:90]}", expanded=i == 1):
            st.write(f"**Claim:** {claim.get('claim','')}")
            st.write(f"**Assessment:** {claim.get('reason') or 'No reasoning returned.'}")
            ids = [int(x) for x in claim.get("evidence_ids", []) if int(x) in by_id]
            if not ids:
                st.caption("No matching fact-check sources were retrieved for this claim.")
                continue
            st.write("**Sources:**")
            for cid in ids:
                citation = by_id[cid]
                publisher = citation.get("publisher", "Unknown")
                rating = citation.get("rating", "Unrated")
                url = citation.get("url", "")
                if url:
                    st.markdown(f"- [{cid}] [{publisher}]({url}) — rated *{rating}*")
                else:
                    st.markdown(f"- [{cid}] {publisher} — rated *{rating}* (no public link)")


# ----------------------------------------------------------------------- page
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="vt-hero">
          <h1>VeriTruth</h1>
          <p>Agentic, explainable fake-news investigation —
             a verdict, a calibrated trust score, the words that drove it,
             and cited evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "article" not in st.session_state:
        st.session_state.article = SAMPLES["Fake"]
    if "result" not in st.session_state:
        st.session_state.result = None

    with st.sidebar:
        st.subheader("Backend")
        st.code(API_URL, language="text")
        health = get_health()
        if health is None:
            st.error("API unreachable.")
        else:
            ok = health.get("status") == "ok"
            (st.success if ok else st.warning)(f"status: {health.get('status')}")
            st.write(f"Model: `{health.get('model','?')}`")
            st.write(f"Evidence chunks: {health.get('evidence_chunks', 0)}")
            st.write(f"Gemini configured: {health.get('llm_configured')}")
            st.write(f"Fact Check API: {health.get('factcheck_api_configured')}")
        st.divider()
        st.caption(
            "Highlighting: red pushes the model toward **Fake**, "
            "green toward **Real**. Hover a token for its weight."
        )

    st.markdown('<div class="vt-section">Load a sample</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, text) in zip(cols, SAMPLES.items(), strict=False):
        if col.button(f"{name} example", use_container_width=True):
            st.session_state.article = text
            st.session_state.result = None

    article = st.text_area(
        "Paste a headline or article",
        height=220,
        key="article",
    )

    left, right = st.columns([1, 1])
    run_quick = left.button("Quick check", type="secondary", use_container_width=True)
    run_full = right.button("Full investigation", type="primary", use_container_width=True)

    if run_quick or run_full:
        if not article.strip():
            st.warning("Please enter some text first.")
        elif run_quick:
            with st.spinner("Scoring..."):
                prediction, error = call_api("/predict", {"text": article})
                explanation, _ = call_api("/explain", {"text": article, "top_k": 10})
            if error:
                st.error(error)
            else:
                merged = dict(prediction or {})
                merged["tokens"] = (explanation or {}).get("tokens", [])
                merged["claims"] = []
                merged["citations"] = []
                merged["explanation"] = ""
                st.session_state.result = merged
        else:
            with st.spinner("Investigating: decomposing claims, retrieving evidence..."):
                result, error = call_api("/investigate", {"text": article})
            if error:
                st.error(error)
            else:
                st.session_state.result = result

    result = st.session_state.result
    if not result:
        st.info("Load a sample or paste text, then run a check.")
        return

    render_verdict(
        str(result.get("verdict", "Suspicious")),
        float(result.get("trust_score", 50.0)),
        bool(result.get("degraded", False)),
    )

    if result.get("explanation"):
        st.markdown('<div class="vt-section">Agent verdict</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="vt-card">{html.escape(str(result["explanation"]))}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="vt-section">What the model reacted to</div>', unsafe_allow_html=True
    )
    tokens = result.get("tokens", [])
    st.markdown(highlight_text(article, tokens), unsafe_allow_html=True)
    if tokens:
        with st.expander("Token weights table"):
            st.dataframe(
                [
                    {"token": t.get("word"), "weight": round(float(t.get("weight", 0)), 5)}
                    for t in tokens
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("No token attributions were available for this text.")

    if result.get("claims") is not None and (result.get("claims") or result.get("citations")):
        st.markdown(
            '<div class="vt-section">Claims and evidence</div>', unsafe_allow_html=True
        )
        render_claims(result.get("claims", []), result.get("citations", []))

    with st.expander("Was this verdict wrong? Send feedback"):
        corrected = st.radio(
            "What do you believe the correct verdict is?",
            ["Real", "Suspicious", "Fake"],
            horizontal=True,
        )
        comment = st.text_input("Optional comment")
        if st.button("Submit feedback"):
            payload = {
                "text": article,
                "predicted_verdict": result.get("verdict", "Suspicious"),
                "user_verdict": corrected,
                "trust_score": float(result.get("trust_score", 50.0)),
                "comment": comment,
            }
            saved, error = call_api("/feedback", payload)
            if error:
                st.error(error)
            elif saved and saved.get("stored"):
                st.success(f"Thanks — feedback #{saved.get('id')} stored.")
            else:
                st.warning("Feedback could not be stored; the backend logged the error.")

    with st.expander("Raw response"):
        st.json(result)


if __name__ == "__main__":
    main()
