"""Streamlit frontend for VeritasCheck.

Layout (in the order required by the specification):

1. Verdict card
2. Human-readable explanation
3. Claim breakdown
4. Source Provenance and Evidence   <-- the transparency section
5. User-Submitted Article or Clip
6. ML model assessment
7. Limitations and recommended next action

The frontend holds no secrets. It calls the FastAPI backend, which owns the keys.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Allow `streamlit run frontend/streamlit_app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.provenance import (  # noqa: E402
    NEWS_NOTICE,
    NO_SOURCES_NOTICE,
    SEARCH_FAILED_NOTICE,
)
from app.schemas import MISSING  # noqa: E402

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 120

VERDICT_STYLES = {
    "Likely Real": ("✅", "#0f5132", "#d1e7dd"),
    "Likely Fake": ("⛔", "#842029", "#f8d7da"),
    "Needs Verification": ("⚠️", "#664d03", "#fff3cd"),
}

RELATION_BADGES = {
    "SUPPORTS": "🟢 Supports the claim",
    "CONTRADICTS": "🔴 Contradicts the claim",
    "PARTIALLY_SUPPORTS": "🟡 Partially supports the claim",
    "UNRELATED": "⚪ Unrelated to the claim",
    "UNKNOWN": "❔ Unable to assess",
}

EVIDENCE_LABELS = {
    "RELEVANT": "Relevant evidence",
    "WEAK": "Weak or partially relevant evidence",
    "CONTRADICTORY": "Contradictory evidence",
    "UNRELATED": "Unrelated result",
    "UNKNOWN": "Unable to assess",
}

CLAIM_BADGES = {
    "SUPPORTED": "🟢 Supported",
    "CONTRADICTED": "🔴 Contradicted",
    "PARTIALLY_SUPPORTED": "🟡 Partially supported",
    "UNVERIFIED": "⚪ Unverified",
}

INPUT_TYPE_LABELS = {
    "HEADLINE": "Looks like a headline",
    "ARTICLE_CLIP": "Looks like a paragraph or article clipping",
    "FULL_ARTICLE": "Looks like a full article",
    "UNKNOWN": "Could not determine the shape of this input",
}


st.set_page_config(page_title="TruthLens AI", page_icon="📰", layout="wide")


def inject_theme() -> None:
    paper_image = (
        Path(__file__).resolve().parent.parent
        / "old-grunge-news-paper-background-black-white-grungy-paper-texture-vintage-newsprint-design-scratched-poster-template_1028938-314871.avif"
    )
    paper_url = (
        "data:image/avif;base64,"
        + base64.b64encode(paper_image.read_bytes()).decode("utf-8")
    )

    st.markdown(
        f"""
        <style>
        :root {{
            --truthlens-bg: #071827;
            --truthlens-panel: rgba(10, 19, 31, 0.68);
            --truthlens-panel-strong: rgba(9, 16, 28, 0.82);
            --truthlens-border: rgba(148, 163, 184, 0.28);
            --truthlens-blue: #64b5ff;
            --truthlens-blue-strong: #2d7ef7;
            --truthlens-text: #edf5ff;
            --truthlens-soft: #dce7f7;
            --truthlens-muted: rgba(220, 231, 247, 0.8);
        }}

        html, body {{
            background: linear-gradient(180deg, rgba(7, 24, 39, 0.9), rgba(7, 24, 39, 0.96));
        }}

        .stApp {{
            background: linear-gradient(180deg, rgba(7, 24, 39, 0.78), rgba(7, 24, 39, 0.88));
            color: var(--truthlens-text);
        }}

        [data-testid="stHeader"] {{
            background: rgba(7, 24, 39, 0.2);
            box-shadow: none;
            border-bottom: 0;
        }}

        .block-container {{
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }}

        .stApp {{
            background: #071426;
        }}

        .block-container {{
            max-width: 100% !important;
            padding: 0 !important;
        }}

        .truthlens-shell {{
            position: relative;
            min-height: 100vh;
            min-height: 100svh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            border: 0;
            border-radius: 0;
            overflow: hidden;
            box-shadow: none;
            background: rgba(11, 18, 29, 0.55);
        }}

        .truthlens-shell::before {{
            content: "";
            position: absolute;
            inset: 0;
            background-image: linear-gradient(rgba(11, 17, 28, 0.6), rgba(11, 17, 28, 0.75)),
                url("{paper_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: grayscale(1) contrast(1.25) brightness(0.65);
            opacity: 0.9;
        }}

        .truthlens-card {{
            position: relative;
            z-index: 1;
            text-align: center;
            width: min(100%, 1100px);
            padding: 2.2rem 1rem 2.5rem;
        }}

        .truthlens-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1.25rem;
            padding: 0.55rem 1.2rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.8);
            color: #0d2f4f;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}

        .truthlens-badge-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--truthlens-blue-strong);
            box-shadow: 0 0 0 4px rgba(45, 126, 247, 0.15);
        }}

        .truthlens-title {{
            margin: 0;
            font-size: clamp(4rem, 9vw, 10rem);
            line-height: 0.9;
            letter-spacing: -0.06em;
            font-weight: 900;
            color: #fff;
            text-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        }}

        .truthlens-title .highlight {{
            font-weight: 800;
            color: rgba(255, 255, 255, 0.95);
        }}

        .truthlens-quote {{
            margin-top: 1.5rem;
            font-size: clamp(1.65rem, 2vw, 2.5rem);
            font-family: Georgia, 'Times New Roman', serif;
            color: rgba(255, 255, 255, 0.96);
            font-style: italic;
            font-weight: 600;
            line-height: 1.3;
        }}

        .truthlens-subquote {{
            margin-top: 1rem;
            font-size: clamp(1.1rem, 1.4vw, 1.6rem);
            font-family: Georgia, 'Times New Roman', serif;
            color: rgba(255, 255, 255, 0.8);
            font-style: italic;
        }}

        div[data-testid="stButton"] button[kind="primary"] {{
            display: block;
            min-height: 52px;
            min-width: 320px;
            margin: 2rem auto 0;
            border-radius: 28px;
            background: #0b172d;
            border: 1px solid rgba(148, 163, 184, 0.45);
            color: #ffffff;
            font-weight: 700;
            font-size: 1rem;
            box-shadow: 0 14px 26px rgba(1, 8, 22, 0.36);
        }}

        div[data-testid="stButton"] button[kind="primary"]:hover {{
            background: #112340;
            border-color: rgba(148, 163, 184, 0.7);
        }}

        .truthlens-form {{
            position: relative;
            z-index: 2;
            background: rgba(7, 24, 39, 0.76);
            border: 1px solid var(--truthlens-border);
            border-radius: 26px;
            padding: 1.25rem 1.25rem 0.5rem;
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.4);
            backdrop-filter: blur(2px);
        }}

        .truthlens-form .stTextArea textarea {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            color: #edf5ff;
            min-height: 160px !important;
            font-size: 1rem;
            resize: vertical;
        }}

        .truthlens-form .stTextArea textarea:focus {{
            border-color: rgba(100, 181, 255, 0.9);
            box-shadow: 0 0 0 1px rgba(100, 181, 255, 0.35);
        }}

        .truthlens-form .stTextArea label {{
            color: rgba(237, 245, 255, 0.9);
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}

        .truthlens-demo-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
            color: var(--truthlens-soft);
        }}

        .truthlens-status-box {{
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: rgba(19, 33, 48, 0.7);
            border: 1px solid rgba(156, 163, 175, 0.2);
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .truthlens-status-box .dot {{
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: #4ade80;
        }}

        .truthlens-status-box .offline {{
            background: #fbbf24;
        }}

        .truthlens-main-btn button {{
            width: 100%;
            background: linear-gradient(135deg, #2d7ef7, #64b5ff);
            color: white;
            border: none;
            border-radius: 16px;
            font-size: 1rem;
            font-weight: 700;
            padding: 0.95rem 1.2rem;
            box-shadow: 0 10px 22px rgba(45, 126, 247, 0.35);
        }}

        .truthlens-main-btn button:hover {{
            filter: brightness(1.04);
        }}

        .truthlens-main-btn button:focus {{
            box-shadow: 0 0 0 2px rgba(164, 201, 255, 0.35);
        }}

        .truthlens-result {{
            background: rgba(8, 15, 26, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 20px;
            padding: 1.25rem 1.2rem;
            margin-top: 1.5rem;
        }}

        .truthlens-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.9rem 1.1rem;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.5rem;
        }}

        .truthlens-nav-brand {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-size: 1.05rem;
            font-weight: 800;
            color: #0f172a;
        }}

        .truthlens-nav-logo {{
            width: 22px;
            height: 22px;
            border-radius: 8px;
            background: linear-gradient(135deg, #2d7ef7, #64b5ff);
            box-shadow: 0 5px 12px rgba(45, 126, 247, 0.28);
        }}

        .truthlens-nav-actions {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
        }}

        .truthlens-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(255, 255, 255, 0.7);
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 600;
        }}

        .truthlens-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            display: inline-block;
        }}

        .truthlens-page-card {{
            position: relative;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1.4rem 1.4rem 0.2rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
        }}

        .truthlens-hero-title {{
            text-align: center;
            font-size: clamp(2.6rem, 5vw, 5rem);
            line-height: 1.05;
            letter-spacing: -0.05em;
            margin: 0.5rem 0 0.4rem;
            font-weight: 900;
            color: #0f172a;
        }}

        .truthlens-hero-kicker {{
            text-align: center;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #2d7ef7;
            font-weight: 800;
        }}

        .truthlens-hero-copy {{
            text-align: center;
            max-width: 860px;
            margin: 0.4rem auto 1.2rem;
            font-size: clamp(1.1rem, 2vw, 1.4rem);
            color: #334155;
            line-height: 1.5;
        }}

        .truthlens-input-panel {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.3rem 1.3rem 0.7rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }}

        @media (max-width: 700px) {{
            .truthlens-shell {{
                min-height: 540px;
                margin: 0 -0.5rem 1.5rem;
            }}

            .truthlens-cta {{
                min-width: 0;
                width: 100%;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- helpers ---------------------------------------------------------------


def show_value(value: str | None) -> str:
    """Render a field, using the explicit sentinel when data is absent."""
    if value is None:
        return MISSING
    value = str(value).strip()
    return value if value else MISSING


def call_backend(text: str, max_sources: int) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "max_sources": max_sources},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return None, "The analysis request timed out. Please try again."
    except requests.exceptions.RequestException:
        return None, (
            f"Could not reach the backend at {BACKEND_URL}. Start it with "
            "`uvicorn app.main:app --reload`."
        )
    if response.status_code != 200:
        return None, (
            f"The backend returned an error (HTTP {response.status_code}). "
            "Please try again."
        )
    try:
        return response.json(), None
    except ValueError:
        return None, "The backend returned an unreadable response."


def backend_health() -> dict | None:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None


# --- sections --------------------------------------------------------------


def render_verdict(analysis: dict) -> None:
    verdict = analysis.get("verdict", "Needs Verification")
    confidence = analysis.get("confidence", 0)
    icon, fg, bg = VERDICT_STYLES.get(verdict, VERDICT_STYLES["Needs Verification"])

    st.markdown(
        f"""
        <div style="background:{bg};border-left:8px solid {fg};padding:20px;
                    border-radius:8px;margin-bottom:8px;">
          <div style="color:{fg};font-size:30px;font-weight:700;">
            {icon} {verdict}
          </div>
          <div style="color:{fg};font-size:17px;margin-top:6px;">
            Confidence: {confidence}/100
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(int(confidence), 0), 100) / 100)

    summary = analysis.get("headline_summary", "")
    if summary:
        st.caption(f"Analysed statement: {summary}")

    generated_by = analysis.get("generated_by", "DETERMINISTIC_FALLBACK")
    if generated_by == "DETERMINISTIC_FALLBACK":
        st.info(
            "This explanation was produced by the built-in rule engine, not by "
            "the AI model. The verdict and the evidence below are unaffected."
        )


def render_explanation(analysis: dict) -> None:
    st.subheader("What this means")
    st.write(
        analysis.get("plain_language_explanation")
        or "No explanation could be generated."
    )
    agreement = analysis.get("source_agreement", "NONE")
    st.caption(f"Agreement between retrieved sources: **{agreement}**")


def render_claims(analysis: dict, sources_by_id: dict) -> None:
    st.subheader("Claim breakdown")
    breakdown = analysis.get("claim_breakdown") or []
    if not breakdown:
        st.write("No individual claims were extracted from this input.")
        return

    for item in breakdown:
        badge = CLAIM_BADGES.get(item.get("status", "UNVERIFIED"), "⚪ Unverified")
        claim_id = item.get("claim_id", "")
        with st.container(border=True):
            st.markdown(f"**{claim_id} — {badge}**")
            st.write(item.get("claim_text", ""))
            if item.get("explanation"):
                st.caption(item["explanation"])

            source_ids = item.get("source_ids") or []
            if source_ids:
                st.markdown("Cited sources: " + ", ".join(f"`{s}`" for s in source_ids))
                for source_id in source_ids:
                    source = sources_by_id.get(source_id)
                    if not source:
                        continue
                    with st.expander(
                        f"{source_id} — {show_value(source.get('publisher'))}"
                    ):
                        render_source_detail(source)
            else:
                st.caption(
                    "No retrieved article was close enough to this statement to "
                    "be cited. That is not evidence that the statement is false."
                )


def render_source_detail(source: dict) -> None:
    """Full metadata for one News API article."""
    st.markdown(f"**Source ID:** `{source.get('source_id', '')}`")
    st.markdown(f"**Source type:** `{source.get('source_type', 'NEWS_API_RESULT')}`")
    st.markdown(f"**Publisher:** {show_value(source.get('publisher'))}")
    st.markdown(f"**Title:** {show_value(source.get('title'))}")

    url = show_value(source.get("url"))
    if url != MISSING:
        # URLs are always shown in full and are always clickable.
        st.markdown(f"**Article URL:** [{url}]({url})")
    else:
        st.markdown(f"**Article URL:** {MISSING}")

    st.markdown(f"**Author:** {show_value(source.get('author'))}")
    st.markdown(f"**Published at:** {show_value(source.get('published_at'))}")
    st.markdown(f"**Description / excerpt:** {show_value(source.get('description'))}")
    st.markdown(f"**Search query that retrieved it:** `{show_value(source.get('retrieval_query'))}`")
    st.markdown(f"**Retrieved at:** {show_value(source.get('retrieved_at'))}")

    col1, col2 = st.columns(2)
    col1.metric("Text similarity", f"{source.get('text_similarity', 0.0):.2f}")
    col2.metric("Relevance score", f"{source.get('relevance_score', 0.0):.2f}")

    relation = source.get("claim_relation", "UNKNOWN")
    evidence = source.get("evidence_status", "UNKNOWN")
    st.markdown(f"**Relation to your claim:** {RELATION_BADGES.get(relation, relation)}")
    st.markdown(
        f"**Reliability status:** {EVIDENCE_LABELS.get(evidence, 'Unable to assess')}"
    )
    used = "Yes" if source.get("used_in_final_answer") else "No"
    st.markdown(f"**Used in final answer:** {used}")
    st.caption(show_value(source.get("source_quality_hint")))


def render_provenance(result: dict) -> None:
    st.subheader("Source Provenance and Evidence")
    st.caption(NEWS_NOTICE)

    provenance = result.get("source_provenance") or {}
    news_search = result.get("news_search") or {}
    sources = provenance.get("news_api_sources") or []

    # Search queries actually issued.
    queries = news_search.get("queries") or []
    if queries:
        with st.expander(f"Search queries used ({len(queries)})", expanded=False):
            for query in queries:
                st.markdown(
                    f"- `{query.get('query_id')}` **{query.get('query_type')}**: "
                    f"{query.get('query_text')}"
                )

    if not news_search.get("ok", True):
        st.error(news_search.get("error") or SEARCH_FAILED_NOTICE)
    elif not sources:
        st.warning(NO_SOURCES_NOTICE)

    if sources:
        used_count = sum(1 for s in sources if s.get("used_in_final_answer"))
        st.markdown(
            f"**{len(sources)} article(s) retrieved — {used_count} used as "
            "evidence in the final answer.**"
        )
        for source in sources:
            relation = source.get("claim_relation", "UNKNOWN")
            badge = RELATION_BADGES.get(relation, relation)
            used_mark = "★ used" if source.get("used_in_final_answer") else "not used"
            header = (
                f"{source.get('source_id')} · {show_value(source.get('publisher'))} · "
                f"{badge} · {used_mark}"
            )
            with st.expander(header, expanded=False):
                st.markdown(f"### {show_value(source.get('title'))}")
                render_source_detail(source)

    # The two non-external source types, kept visually distinct.
    st.markdown("---")
    st.markdown("**Non-evidence sources used to produce this page**")

    model_source = provenance.get("model_source") or {}
    ai_source = provenance.get("ai_explanation_source") or {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"`{model_source.get('source_id', 'MODEL-001')}` — "
            f"**{model_source.get('source_type', 'MODEL_OUTPUT')}**"
        )
        st.caption(model_source.get("description", ""))
        st.caption("This is not an external source and is never proof.")
    with col2:
        st.markdown(
            f"`{ai_source.get('source_id', 'AI-001')}` — "
            f"**{ai_source.get('source_type', 'AI_EXPLANATION')}**"
        )
        st.caption(ai_source.get("description", ""))
        st.caption(
            "This is an interpretation of the evidence above, not an "
            "independent source."
        )


def render_user_clip(result: dict) -> None:
    st.subheader("User-Submitted Article or Clip")
    provenance = result.get("source_provenance") or {}
    user_source = provenance.get("user_submitted_source") or {}

    st.markdown(f"**Source ID:** `{user_source.get('source_id', 'USER-001')}`")
    st.markdown(f"**Source type:** `{user_source.get('source_type', 'USER_SUBMITTED_TEXT')}`")
    st.markdown(f"**Character count:** {user_source.get('character_count', 0)}")
    st.markdown(
        "**Input shape:** "
        + INPUT_TYPE_LABELS.get(user_source.get("input_type", "UNKNOWN"), "Unknown")
    )

    urls = user_source.get("user_supplied_urls") or []
    if urls:
        st.markdown("**URL supplied by user**")
        for url in urls:
            st.markdown(f"- [{url}]({url})")
        st.caption(
            "This URL was taken from the text you pasted. It has not been "
            "opened or checked, and it is not assumed to be genuine or to "
            "support the claim."
        )

    with st.expander("Show the exact text that was analysed", expanded=False):
        st.text(user_source.get("text", ""))
        if user_source.get("truncated"):
            st.caption("This display was truncated for length.")

    st.warning(
        user_source.get(
            "notice",
            "This text was supplied by the user. It was not independently "
            "verified and may be incomplete or edited.",
        )
    )


def render_ml(result: dict) -> None:
    st.subheader("Machine-learning model assessment")
    ml = result.get("ml_result") or {}

    if not ml.get("available", False):
        st.info(
            ml.get("note")
            or "The machine-learning ensemble was not available for this request."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble prediction", ml.get("prediction", "UNKNOWN"))
    col2.metric("Ensemble confidence", f"{ml.get('confidence', 0)}%")
    col3.metric("Members agree", "Yes" if ml.get("models_agree", True) else "No")

    votes = ml.get("votes") or []
    if votes:
        st.markdown("**Individual model votes**")
        st.table(
            [
                {
                    "Model": v.get("model_name", ""),
                    "Prediction": v.get("prediction", ""),
                    "Confidence": f"{v.get('confidence', 0)}%",
                }
                for v in votes
            ]
        )

    if ml.get("note"):
        st.warning(ml["note"])

    st.caption(
        "MODEL-001 · MODEL_OUTPUT — this is a writing-style signal, not "
        "external evidence, and it is never treated as proof."
    )


def render_limitations(result: dict) -> None:
    analysis = result.get("final_analysis") or {}
    st.subheader("Recommended next step")
    st.info(analysis.get("recommended_action") or "Verify with independent sources.")

    limitations = analysis.get("limitations") or []
    if limitations:
        st.subheader("Limitations of this analysis")
        for limitation in limitations:
            st.markdown(f"- {limitation}")

    warnings = result.get("system_warnings") or []
    if warnings:
        with st.expander(f"System warnings ({len(warnings)})", expanded=False):
            for warning in warnings:
                st.markdown(f"- {warning}")


# --- page ------------------------------------------------------------------


def render_landing_page() -> None:
    st.markdown(
        """
        <div class="truthlens-shell">
            <div class="truthlens-card">
                <div class="truthlens-badge">
                    <span class="truthlens-badge-dot"></span>
                    <span>AI-Powered News Verification</span>
                </div>
                <h1 class="truthlens-title">TruthLens <span class="highlight">AI</span></h1>
                <div class="truthlens-quote">“See Beyond the Headlines. Discover the Truth.”</div>
                <div class="truthlens-subquote">“Because Every Story Deserves the Truth.”</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        if st.button(
            "Click anywhere on screen to enter →",
            key="landing_enter",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "verification"
            st.rerun()


def render_verification_page() -> None:
    st.markdown(
        """
        <div class="truthlens-nav">
            <div class="truthlens-nav-brand">
                <div class="truthlens-nav-logo"></div>
                <span>TruthLens AI</span>
            </div>
            <div class="truthlens-nav-actions">
                <div class="truthlens-pill">Front Page</div>
                <div class="truthlens-pill">API Specs</div>
                <div class="truthlens-pill"><span class="truthlens-dot"></span>API Connected</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="truthlens-hero-kicker">Story &amp; Claim Verification</div>
        <h1 class="truthlens-hero-title">Verify Before You Share.</h1>
        <p class="truthlens-hero-copy">Paste any news story, social media clip, or claim to evaluate credibility signals, examine influential keywords, and check cited evidence.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="truthlens-input-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        text = st.text_area(
            "Article or Claim Input",
            value=st.session_state.get("input_text", ""),
            height=180,
            placeholder="Paste the news story, headline, or claim you want to verify...",
            key="input_text_area",
            label_visibility="visible",
        )
    with col2:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        max_sources = st.slider("Sources", 3, 30, 10, key="max_sources")
        st.caption("Deep source retrieval and evidence checking")

    actions = st.columns([1, 1])
    with actions[0]:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("result", None)
            st.session_state["input_text"] = ""
            st.session_state.pop("input_text_area", None)
    with actions[1]:
        if st.button("Analyze & Verify Story", type="primary", use_container_width=True):
            if not text.strip():
                st.error("Please enter a headline or article to analyse.")
            else:
                st.session_state["input_text"] = text
                with st.spinner("Classifying, searching for sources and checking claims…"):
                    result, error = call_backend(text, max_sources)
                if error:
                    st.error(error)
                    st.session_state.pop("result", None)
                elif result is None:
                    st.error("No result was returned.")
                    st.session_state.pop("result", None)
                else:
                    st.session_state["result"] = result
                    st.session_state["analysis_result"] = result
                    st.session_state.page = "result"
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_page() -> None:
    st.markdown(
        """
        <div class="truthlens-nav">
            <div class="truthlens-nav-brand">
                <div class="truthlens-nav-logo"></div>
                <span>TruthLens AI</span>
            </div>
            <div class="truthlens-nav-actions">
                <button type="button" style="background: rgba(255,255,255,0.8); border:1px solid rgba(148,163,184,0.35); border-radius:12px; padding:0.4rem 0.8rem; font-weight:600; color:#0f172a; cursor:pointer;">← Verify Another Story</button>
                <div class="truthlens-pill">API Specs</div>
                <div class="truthlens-pill"><span class="truthlens-dot"></span>API Connected</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Verify Another Story", key="verify_another"):
        st.session_state.analysis_result = None
        st.session_state.article_text = ""
        st.session_state.page = "verification"
        st.rerun()

    result = st.session_state.get("analysis_result") or st.session_state.get("result")
    if not result:
        st.info("No verification result available yet.")
        return

    sources_by_id = {
        s.get("source_id"): s
        for s in (result.get("source_provenance") or {}).get("news_api_sources", [])
    }

    analysis = result.get("final_analysis") or {}

    st.markdown('<div class="truthlens-result">', unsafe_allow_html=True)
    render_verdict(analysis)
    render_explanation(analysis)
    st.markdown("---")
    render_claims(analysis, sources_by_id)
    st.markdown("---")
    render_provenance(result)
    st.markdown("---")
    render_user_clip(result)
    st.markdown("---")
    render_ml(result)
    st.markdown("---")
    render_limitations(result)

    with st.expander("Raw backend response (JSON)", expanded=False):
        st.json(result)

    st.caption(
        f"Request ID: {result.get('request_id', '')} · "
        f"Analysed at: {result.get('analyzed_at', '')}"
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_banner() -> None:
    health = backend_health()
    status_text = "Backend online" if health else "Backend check pending"
    status_class = "dot" if health else "dot offline"
    with st.container():
        st.markdown(
            f"""
            <div class="truthlens-demo-bar">
                <div class="truthlens-status-box"><span class="{status_class}"></span>{status_text}</div>
                <div class="truthlens-status-box"><span class="dot"></span>{BACKEND_URL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_theme()

    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""

    page = st.session_state.page

    if page == "landing":
        render_landing_page()
    elif page == "verification":
        render_verification_page()
    elif page == "result":
        render_result_page()


if __name__ == "__main__":
    main()
