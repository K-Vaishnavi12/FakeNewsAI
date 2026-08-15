// VeriTruth / TruthLens AI — FastAPI Backend Integration Client
// Base URL for FastAPI: http://localhost:8000
//
// INTEGRATION NOTE (added during backend integration):
// The original client called /predict, /explain and /investigate. The live
// VeritasCheck backend exposes a single richer endpoint, POST /analyze, which
// returns the classification, the retrieved News API evidence, the claim
// decomposition and the full source provenance in one response.
//
// `verifyArticle` now calls /analyze and maps that response onto the exact
// shape this interface already expects, so no UI code had to be rewritten.
// The raw backend payload is preserved on `data.provenance` so the added
// provenance sections can render real, traceable evidence.
//
// Documented Endpoints in use: /health, /analyze, /feedback

import { DEMO_SAMPLES } from './demoData.js';

const API_BASE_URL = 'http://localhost:8000';

// Analysis performs a live news search plus an LLM call, so it needs a much
// longer budget than the original 5 s prediction timeout.
const ANALYZE_TIMEOUT_MS = 120000;

export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(2500)
    });
    if (response.ok) {
      const data = await response.json();
      return { isOnline: true, data };
    }
    return { isOnline: false };
  } catch (err) {
    return { isOnline: false, error: err.message };
  }
}

export async function verifyArticle(text, sampleId = null) {
  // If explicitly requested demo sample (benchmark preset cards)
  if (sampleId) {
    const found = DEMO_SAMPLES.find(s => s.id === sampleId);
    if (found) {
      return { isDemo: true, data: normalizeVerificationData(found) };
    }
  }

  const startedAt = performance.now();

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, max_sources: 10 }),
      signal: AbortSignal.timeout(ANALYZE_TIMEOUT_MS)
    });

    if (!response.ok) {
      throw new Error(
        `FastAPI /analyze returned HTTP ${response.status}.`
      );
    }

    const payload = await response.json();
    if (!payload || typeof payload !== 'object' || !payload.final_analysis) {
      throw new Error('FastAPI /analyze returned an unexpected response shape.');
    }

    const elapsedMs = Math.round(performance.now() - startedAt);

    return {
      isDemo: false,
      data: normalizeVerificationData(mapAnalyzeResponse(payload, text, elapsedMs))
    };
  } catch (error) {
    // Transparent fallback: the interface still renders, but the added
    // "Backend Status" notice makes it explicit that this is NOT an analysis
    // of the user's text.
    console.warn(
      'FastAPI backend unavailable, serving transparent DEMO MODE sample fallback:',
      error.message
    );
    const fallbackSample = DEMO_SAMPLES[0];
    return {
      isDemo: true,
      error:
        'The verification backend could not be reached (' +
        error.message +
        ') Showing a stored benchmark sample. This is NOT an analysis of your ' +
        'submitted text. Start the backend with: uvicorn app.main:app --reload',
      data: normalizeVerificationData({
        ...fallbackSample,
        rawText: text,
        title: 'User Article (DEMO MODE Fallback)'
      })
    };
  }
}

export async function submitFeedback(feedbackPayload) {
  try {
    const res = await fetch(`${API_BASE_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedbackPayload),
      signal: AbortSignal.timeout(3000)
    });
    return { success: res.ok };
  } catch (err) {
    console.warn("Backend /feedback offline. Saved locally for demonstration.", feedbackPayload);
    return { success: true, isDemo: true };
  }
}

// ---------------------------------------------------------------------------
// Mapping: VeritasCheck /analyze response  ->  TruthLens interface shape
// ---------------------------------------------------------------------------

const CLAIM_STATUS_TEXT = {
  SUPPORTED: 'Supported',
  CONTRADICTED: 'Contradicted',
  PARTIALLY_SUPPORTED: 'Partially Supported',
  UNVERIFIED: 'Unverified'
};

const RELATION_TEXT = {
  SUPPORTS: 'Supporting',
  CONTRADICTS: 'Contradicting',
  PARTIALLY_SUPPORTS: 'Partially Supporting',
  UNRELATED: 'Unrelated',
  UNKNOWN: 'Unassessed'
};

const MISSING = 'Not provided by the source.';

// The interface is built around a 0-100 gauge. It now shows *Verification
// Confidence* - confidence in the verification outcome - not a "truth score".
// The backend supplies this directly, so no re-derivation is needed.
function statusToClass(status) {
  if (status === 'Supported by Retrieved Evidence') return 'real';
  if (status === 'Contradicted by Retrieved Evidence') return 'fake';
  return 'suspicious';
}

function safeText(value) {
  if (value === null || value === undefined) return MISSING;
  const s = String(value).trim();
  return s.length ? s : MISSING;
}

export function mapAnalyzeResponse(payload, submittedText, elapsedMs) {
  const analysis = payload.final_analysis || {};
  const ml = payload.ml_result || {};
  const provenance = payload.source_provenance || {};
  const newsSearch = payload.news_search || {};
  const sources = provenance.news_api_sources || [];
  const scores = payload.verification_scores || {};
  const structured = payload.structured_explanation || {};
  const parsedInput = payload.parsed_input || {};

  // Evidence-first status drives the display; the legacy verdict is a fallback.
  const finalStatus = scores.final_status || analysis.verdict || 'Needs Verification';
  const verificationConfidence =
    scores.verification_confidence ?? analysis.confidence ?? 0;

  // Index sources by ID so each claim can carry its own cited evidence.
  const sourcesById = {};
  sources.forEach(s => { sourcesById[s.source_id] = s; });

  const claims = (analysis.claim_breakdown && analysis.claim_breakdown.length
    ? analysis.claim_breakdown
    : (payload.claims || []).map(c => ({
        claim_id: c.claim_id,
        claim_text: c.claim_text,
        status: c.relation,
        explanation: c.explanation,
        source_ids: c.source_ids
      }))
  ).map(item => {
    const citedIds = item.source_ids || [];
    const cited = citedIds.map(sid => sourcesById[sid]).filter(Boolean);

    // Links must always be visible. When the evidence engine cited few or no
    // sources, the remaining retrieved articles are still listed underneath,
    // clearly labelled with their measured relation so a merely-related or
    // unrelated result is never presented as confirmation.
    const extras = sources
      .filter(s => !citedIds.includes(s.source_id))
      .slice(0, Math.max(0, 4 - cited.length));

    const toEvidence = (s, isCited) => ({
      publisher: safeText(s.publisher),
      title: safeText(s.title),
      snippet: s.description && s.description !== MISSING
        ? s.description
        : (s.title && s.title !== MISSING ? s.title : MISSING),
      relationship: isCited
        ? (RELATION_TEXT[s.claim_relation] || 'Unassessed')
        : (s.claim_relation === 'UNRELATED'
            ? 'Unrelated result — not used as evidence'
            : (RELATION_TEXT[s.claim_relation] || 'Unassessed') + ' (not cited)'),
      url: s.url && s.url !== MISSING ? s.url : '#',
      sourceId: s.source_id,
      relevance: s.relevance_score ?? 0,
      fullTextAvailable: !!s.full_text_available,
      usedAsEvidence: isCited
    });

    return {
      id: item.claim_id,
      text: item.claim_text,
      status: CLAIM_STATUS_TEXT[item.status] || 'Unverified',
      explanation: item.explanation || '',
      evidence: [
        ...cited.map(s => toEvidence(s, true)),
        ...extras.map(s => toEvidence(s, false))
      ]
    };
  });

  // Real linear-model attribution. Empty if the model could not produce it -
  // the UI then states that attribution is unavailable rather than inventing it.
  const shapTokens = (ml.token_attributions || []).map(t => ({
    word: t.word,
    weight: Number(t.weight) || 0
  }));

  return {
    id: payload.request_id || 'live-' + Date.now(),
    title: analysis.headline_summary || (
      submittedText.length > 80 ? submittedText.substring(0, 80) + '...' : submittedText
    ),
    category: 'Live Evidence-Based Verification',
    trustScore: verificationConfidence,
    verdictLabel: finalStatus,
    statusClass: statusToClass(finalStatus),
    summary: structured.what_the_system_found
      || analysis.plain_language_explanation
      || 'No explanation was generated.',
    rawText: submittedText,
    shapTokens,
    explanationMethod: shapTokens.length
      ? (ml.attribution_method || 'Linear model coefficient attribution')
      : 'Attribution unavailable',
    claims,
    // The backend performs no external fact-check lookup, so this stays empty
    // and the corresponding card stays hidden. Nothing is fabricated.
    factChecks: [],
    technicalDetails: {
      model: (ml.votes || []).map(v => v.model_name).join(' + ') || 'Ensemble unavailable',
      modelVersion: ml.model_name || 'VeritasCheck Ensemble',
      processingTimeMs: elapsedMs,
      ragSourcesChecked: sources.length
    },
    // Full backend payload, used by the added provenance sections.
    provenance: {
      requestId: payload.request_id || '',
      analyzedAt: payload.analyzed_at || '',
      userInput: payload.user_input || {},
      parsedInput: parsedInput,
      scores: {
        mlStyleSignal: scores.ml_style_signal ?? 0,
        mlStyleDirection: scores.ml_style_direction || 'UNKNOWN',
        evidenceRelevance: scores.evidence_relevance ?? 0,
        sourceAgreementScore: scores.source_agreement_score ?? 0,
        verificationConfidence: verificationConfidence,
        finalStatus: finalStatus,
        relevantSourceCount: scores.relevant_source_count ?? 0,
        independentPublisherCount: scores.independent_publisher_count ?? 0,
        mlDisclaimer: scores.ml_disclaimer || ''
      },
      structured: {
        verdict: structured.verdict || finalStatus,
        verificationConfidence: structured.verification_confidence ?? verificationConfidence,
        whatTheSystemFound: structured.what_the_system_found || '',
        mlTextPatternSignal: structured.ml_text_pattern_signal || '',
        importantLimitation: structured.important_limitation || '',
        sourceSearch: structured.source_search || '',
        recommendedNextStep: structured.recommended_next_step || ''
      },
      userSubmittedSource: provenance.user_submitted_source || {},
      newsApiSources: sources,
      modelSource: provenance.model_source || {},
      aiExplanationSource: provenance.ai_explanation_source || {},
      newsSearch: {
        ok: newsSearch.ok !== false,
        error: newsSearch.error || null,
        queries: newsSearch.queries || []
      },
      sourceAgreement: analysis.source_agreement || 'NONE',
      recommendedAction: analysis.recommended_action || '',
      limitations: analysis.limitations || [],
      systemWarnings: payload.system_warnings || [],
      generatedBy: analysis.generated_by || '',
      ml: {
        available: ml.available !== false,
        prediction: ml.prediction || 'UNKNOWN',
        confidence: ml.confidence ?? 0,
        modelsAgree: ml.models_agree !== false,
        votes: ml.votes || [],
        note: ml.note || '',
        interpretation: ml.interpretation || 'ML output is a signal, not proof.'
      }
    }
  };
}

// Data Normalizer: Ensures consistent structure across live API & demo samples
export function normalizeVerificationData(item) {
  const trustScore = item.trustScore ?? 50;

  // Demo samples carry no explicit verdict, so they keep the original
  // score-derived labelling. Live results supply the backend's own verdict,
  // which takes precedence so the label can never contradict the analysis.
  let verdictLabel = item.verdictLabel;
  let statusClass = item.statusClass;

  if (!verdictLabel) {
    verdictLabel = 'Suspicious';
    statusClass = 'suspicious';
    if (trustScore >= 70) {
      verdictLabel = 'Likely Real';
      statusClass = 'real';
    } else if (trustScore < 40) {
      verdictLabel = 'Likely Fake';
      statusClass = 'fake';
    }
  }

  return {
    id: item.id || 'res-' + Math.random().toString(36).substr(2, 9),
    title: item.title || 'Submitted Article',
    category: item.category || 'General News',
    trustScore,
    verdictLabel,
    statusClass,
    summary: item.summary || 'Credibility assessment derived from linguistic patterns and verified evidence sources.',
    rawText: item.rawText || '',
    shapTokens: item.shapTokens || item.tokens || [],
    explanationMethod: item.explanationMethod || 'SHAP Feature Attribution',
    claims: (item.claims || []).map((c, idx) => ({
      id: c.id || `claim-${idx + 1}`,
      text: c.text,
      status: c.status || 'Unverified',
      statusClass: c.statusClass || getClaimStatusClass(c.status),
      explanation: c.explanation || '',
      evidence: (c.evidence || []).map(e => ({
        publisher: e.publisher || 'Unknown Publisher',
        title: e.title || 'Referenced Document',
        snippet: e.snippet || 'No excerpt available.',
        relationship: e.relationship || 'Related',
        url: e.url || '#',
        sourceId: e.sourceId || '',
        relevance: e.relevance ?? 0,
        fullTextAvailable: e.fullTextAvailable ?? null,
        usedAsEvidence: e.usedAsEvidence ?? true
      }))
    })),
    factChecks: item.factChecks || [],
    technicalDetails: item.technicalDetails || {
      model: "DistilBERT + RAG Pipeline",
      modelVersion: "1.0.0-demo",
      processingTimeMs: 180,
      ragSourcesChecked: (item.claims || []).reduce((acc, curr) => acc + (curr.evidence?.length || 0), 0)
    },
    // Present only for live results; the added provenance sections check for it.
    provenance: item.provenance || null
  };
}

function getClaimStatusClass(status) {
  if (status === 'Supported') return 'status-supported';
  if (status === 'Contradicted') return 'status-contradicted';
  return 'status-unverified';
}

// Retained from the original client. No longer used for live results: the
// backend now supplies genuine linear-model attribution, so randomised
// placeholder weights are never displayed as if they were model output.
// eslint-disable-next-line no-unused-vars
function generateFallbackTokens(text) {
  const words = text.trim().split(/\s+/);
  return words.slice(0, 40).map(word => ({
    word,
    weight: Math.random() > 0.65 ? (Math.random() * 0.4 + 0.1) : (-Math.random() * 0.4 - 0.1)
  }));
}
