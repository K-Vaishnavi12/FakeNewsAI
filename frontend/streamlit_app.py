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


st.set_page_config(page_title="VeritasCheck", page_icon="🔎", layout="wide")


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


def main() -> None:
    st.title("🔎 VeritasCheck")
    st.caption(
        "Transparent news claim analysis. Every statement below is traceable to "
        "a labelled source."
    )

    with st.sidebar:
        st.header("Status")
        health = backend_health()
        if health is None:
            st.error("Backend unreachable")
            st.caption(f"Expected at {BACKEND_URL}")
            st.code("uvicorn app.main:app --reload", language="bash")
        else:
            st.success("Backend online")
            st.write(
                "News API configured:",
                "✅" if health.get("news_api_configured") else "❌",
            )
            st.write(
                "NVIDIA configured:",
                "✅" if health.get("nvidia_configured") else "❌",
            )
            st.write(
                "ML models loaded:",
                "✅" if health.get("ml_models_available") else "❌",
            )
            if not health.get("ml_models_available"):
                st.code("python -m app.ml.train", language="bash")

        st.header("Options")
        max_sources = st.slider("Maximum sources to retrieve", 3, 30, 10)

        st.header("How to read this")
        st.caption(
            "- Only NEWS_API_RESULT records are external evidence.\n"
            "- MODEL_OUTPUT is a style signal, never proof.\n"
            "- AI_EXPLANATION interprets evidence, it is not a source.\n"
            "- 'Needs Verification' means the evidence was not sufficient, "
            "not that the claim is false."
        )

    text = st.text_area(
        "Paste a news headline, an article clipping, or a full article",
        height=220,
        placeholder="e.g. Officials in Hyderabad approved new funding for regional rail expansion...",
    )

    if st.button("Analyse", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Please enter a headline or article to analyse.")
            return

        with st.spinner("Classifying, searching for sources and checking claims…"):
            result, error = call_backend(text, max_sources)

        if error:
            st.error(error)
            return
        if result is None:
            st.error("No result was returned.")
            return

        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    sources_by_id = {
        s.get("source_id"): s
        for s in (result.get("source_provenance") or {}).get("news_api_sources", [])
    }

    analysis = result.get("final_analysis") or {}

    # 1. Verdict
    render_verdict(analysis)
    # 2. Explanation
    render_explanation(analysis)
    st.markdown("---")
    # 3. Claim breakdown
    render_claims(analysis, sources_by_id)
    st.markdown("---")
    # 4. Source provenance
    render_provenance(result)
    st.markdown("---")
    # 5. User-submitted clip
    render_user_clip(result)
    st.markdown("---")
    # 6. ML assessment
    render_ml(result)
    st.markdown("---")
    # 7. Limitations and next action
    render_limitations(result)

    with st.expander("Raw backend response (JSON)", expanded=False):
        st.json(result)

    st.caption(
        f"Request ID: {result.get('request_id', '')} · "
        f"Analysed at: {result.get('analyzed_at', '')}"
    )


if __name__ == "__main__":
    main()
