import './style.css';
import { checkBackendHealth, verifyArticle, submitFeedback } from './api.js';
import { DEMO_SAMPLES } from './demoData.js';

// Application State Management
const state = {
  currentSlide: 0, // 0: Launch Screen, 1: Verification Desk, 2: Report Dashboard
  inputText: '',
  inputType: 'Article', // 'Article' | 'Claim' | 'Social'
  isBackendOnline: false,
  isLoading: false,
  currentResult: null,
  isDemoMode: false,
  validationError: '',
  feedbackSubmitted: false,
  showTechDetails: false
};

// Smooth Horizontal Slide Navigation
function goToSlide(slideIndex) {
  state.currentSlide = slideIndex;
  const track = document.getElementById('app-slider-track');
  if (track) {
    track.style.transform = `translateX(-${slideIndex * 100}vw)`;
  }
}

// Initialize Application
async function init() {
  renderApp();
  goToSlide(0);

  // Check Backend Health
  const health = await checkBackendHealth();
  state.isBackendOnline = health.isOnline;
  updateStatusBadges();
}

function updateStatusBadges() {
  document.querySelectorAll('.status-badge-container').forEach(container => {
    let statusBadgeText = state.isBackendOnline ? 'API Connected' : 'Demo Benchmarks';
    let statusClass = state.isBackendOnline ? 'online' : 'offline';
    container.innerHTML = `
      <div class="status-badge" title="Verification API Connectivity">
        <span class="status-dot ${statusClass}"></span>
        <span>${statusBadgeText}</span>
      </div>
    `;
  });
}

function renderApp() {
  const app = document.querySelector('#app');
  app.innerHTML = `
    <div class="app-slider-container">
      <div class="app-slider-track" id="app-slider-track">
        
        <!-- SLIDE 0: CLEAN FIRST PAGE / LAUNCH SCREEN -->
        <section class="slide-view" id="slide-splash">
          <div class="splash-container">
            
            <div class="splash-badge-tag">
              <span class="splash-badge-dot"></span>
              <span>AI-Powered News Verification</span>
            </div>

            <h1 class="splash-title-text">TruthLens AI</h1>

            <div class="splash-headline-primary">
              “See Beyond the Headlines. Discover the Truth.”
            </div>

            <div class="splash-headline-secondary">
              “Because Every Story Deserves the Truth.”
            </div>

            <div class="splash-enter-btn" id="splash-launch-btn">
              <span>Click anywhere on screen to enter</span>
              <span>➔</span>
            </div>

          </div>
        </section>

        <!-- SLIDE 1: MAIN INPUT & VERIFICATION DESK -->
        <section class="slide-view" id="slide-portal">
          <div class="main-portal-view">
            ${renderNavbar('portal')}

            <main class="portal-container">
              <div class="hero-heading-block">
                <div class="hero-tag">Story &amp; Claim Verification</div>
                <h1 class="hero-title">Verify Before You Share.</h1>
                <p class="hero-desc">
                  Paste any news story, social media clip, or claim to evaluate credibility signals, examine influential keywords, and check cited evidence.
                </p>
              </div>

              <!-- Input Card -->
              <div class="content-card">
                <div class="card-header-flex">
                  <div class="card-title">Article or Claim Input</div>
                  <div class="format-tabs">
                    <button class="format-tab ${state.inputType === 'Article' ? 'active' : ''}" data-type="Article">News Article</button>
                    <button class="format-tab ${state.inputType === 'Claim' ? 'active' : ''}" data-type="Claim">Single Claim</button>
                    <button class="format-tab ${state.inputType === 'Social' ? 'active' : ''}" data-type="Social">Social Post</button>
                  </div>
                </div>

                <textarea 
                  id="news-input" 
                  class="input-textarea" 
                  placeholder="Paste the news story, headline, or claim you want to verify..."
                >${state.inputText}</textarea>

                <div class="textarea-footer">
                  <span>🔒 Submissions are analyzed securely for credibility patterns &amp; evidence retrieval.</span>
                  <div><span id="char-count">${state.inputText.length}</span> characters</div>
                </div>

                ${state.validationError ? `<div style="color: #EF4444; font-size: 0.85rem; margin-bottom: 1rem; font-weight: 600;">⚠ ${state.validationError}</div>` : ''}

                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                  <div style="display: flex; gap: 0.5rem;">
                    <button id="btn-clear" class="btn btn-secondary" type="button">Clear</button>
                    <button id="btn-paste" class="btn btn-secondary" type="button">Paste Clipboard</button>
                  </div>
                  <button id="btn-analyze" class="btn btn-primary" type="button">
                    <span>Analyze &amp; Verify Story</span>
                    <span>➔</span>
                  </button>
                </div>

                <!-- Sample Presets -->
                <div class="presets-section">
                  <div class="presets-title">Or test with a benchmark sample:</div>
                  <div class="presets-grid">
                    <div class="preset-card" data-sample="demo-sample-1">
                      <div class="preset-badge" style="color: #10B981;">● Verified Science</div>
                      <div class="preset-title">Quantum Computing Record</div>
                      <div class="preset-desc">Peer-reviewed benchmark with clear citations and neutral academic tone.</div>
                    </div>

                    <div class="preset-card" data-sample="demo-sample-2">
                      <div class="preset-badge" style="color: #F59E0B;">● Suspicious Tech</div>
                      <div class="preset-title">Infinite Battery Claim</div>
                      <div class="preset-desc">Sensational commercial claims lacking verifiable experimental sources.</div>
                    </div>

                    <div class="preset-card" data-sample="demo-sample-3">
                      <div class="preset-badge" style="color: #EF4444;">● Viral Fabricated Claim</div>
                      <div class="preset-title">Miracle Health Solution</div>
                      <div class="preset-desc">Viral health misinformation contradicting established clinical data.</div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Trust Pillars -->
              <div class="pillars-grid">
                <div class="pillar-card">
                  <div class="pillar-num">01. NLP ANALYSIS</div>
                  <div class="pillar-title">Neural Classifier</div>
                  <div class="pillar-body">Fine-tuned DistilBERT models analyze stylistic and credibility patterns across textual structure.</div>
                </div>
                <div class="pillar-card">
                  <div class="pillar-num">02. SHAP XAI</div>
                  <div class="pillar-title">Transparent Attribution</div>
                  <div class="pillar-body">Explainable SHAP token weights clarify exactly which words shaped the assessment.</div>
                </div>
                <div class="pillar-card">
                  <div class="pillar-num">03. RAG EVIDENCE</div>
                  <div class="pillar-title">Source Grounding</div>
                  <div class="pillar-body">Extracted atomic assertions are cross-checked against reliable knowledge records and citations.</div>
                </div>
              </div>
            </main>

            ${renderFooter()}
          </div>
        </section>

        <!-- SLIDE 2: VERIFICATION RESULTS DASHBOARD -->
        <section class="slide-view" id="slide-result">
          <div class="result-portal-view">
            ${renderNavbar('result')}

            <main class="result-container" id="result-content-container">
              ${renderResultDashboard()}
            </main>

            ${renderFooter()}
          </div>
        </section>

      </div>
    </div>
  `;

  bindEvents();
  updateStatusBadges();
}

function renderNavbar(currentView) {
  return `
    <nav class="navbar">
      <div class="nav-brand" data-action="go-splash">
        <div class="nav-logo-box">🔍</div>
        <span>TruthLens AI</span>
      </div>

      <ul class="nav-links">
        ${currentView === 'result' ? `
          <li>
            <button class="nav-btn btn-nav-back" type="button">
              <span>← New Verification</span>
            </button>
          </li>
        ` : `
          <li>
            <button class="nav-btn btn-nav-splash" type="button">
              <span>Front Page</span>
            </button>
          </li>
        `}
        <li>
          <a href="http://localhost:8000/docs" target="_blank" class="nav-btn">API Specs</a>
        </li>
        <li class="status-badge-container"></li>
      </ul>
    </nav>
  `;
}

function renderResultDashboard() {
  if (state.isLoading) {
    return `
      <div class="loading-box">
        <div class="spinner"></div>
        <h2 style="font-size: 1.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">
          Analyzing Article Credibility...
        </h2>
        <p style="color: #64748B; font-size: 0.92rem; max-width: 500px; margin: 0 auto; line-height: 1.6;">
          Decomposing claims, retrieving evidence sources from ChromaDB, and computing SHAP attribution weights...
        </p>
      </div>
    `;
  }

  if (!state.currentResult) {
    return `
      <div class="content-card" style="text-align: center; padding: 4rem 2rem; max-width: 600px; margin: 2rem auto;">
        <h2 style="font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem;">No Verification Result Yet</h2>
        <p style="color: #64748B; margin-bottom: 1.5rem;">Submit an article, headline, or claim to generate an explainable credibility assessment.</p>
        <button class="btn btn-primary btn-nav-back" type="button">← Go to Input Form</button>
      </div>
    `;
  }

  const { isDemo, data } = state.currentResult;
  const score = data.trustScore;
  let verdictClass = data.statusClass || "suspicious";
  let verdictIcon = getVerdictIcon(data.verdictLabel);

  const strokeDashoffset = 345.57 - (345.57 * score) / 100;

  return `
    <div class="result-header-bar">
      <button class="nav-btn btn-nav-back" type="button">
        <span>← Verify Another Story</span>
      </button>

      <div style="font-size: 0.8rem; color: #64748B; font-weight: 600;">
        ${isDemo ? '● Sample Benchmark Evaluation' : '● Live Model Inference'}
      </div>
    </div>

    <!-- Main Verdict Hero -->
    <div class="verdict-hero-card ${verdictClass}">
      <div>
        <div class="verdict-pill ${verdictClass}">
          <span>${verdictIcon}</span>
          <span>${data.verdictLabel}</span>
        </div>
        <p class="verdict-summary">${data.summary}</p>
      </div>

      <div class="gauge-box" title="Verification Confidence (0 - 100). Confidence in the verification outcome, not a probability that the claim is true.">
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="55" />
          <circle class="gauge-fill ${verdictClass}" cx="60" cy="60" r="55" 
            stroke-dasharray="345.57" 
            stroke-dashoffset="${strokeDashoffset}" 
            transform="rotate(-90 60 60)" />
        </svg>
        <div class="gauge-center-text">
          <div class="gauge-score-number">${score}</div>
          <div class="gauge-score-label">Verification<br>Confidence</div>
        </div>
      </div>
    </div>

    ${renderScoreBreakdown(data)}

    <!-- 2 Columns in Result -->
    <div class="results-grid-2col">
      
      <!-- Left Column: SHAP Model Influence -->
      <div class="content-card">
        <div class="card-header-flex">
          <span class="card-title">Model Attribution &amp; Keywords</span>
          <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">${data.explanationMethod}</span>
        </div>
        <p style="font-size: 0.84rem; color: #64748B; margin-bottom: 0.85rem;">
          Highlighted words show linguistic patterns that influenced the model prediction.
        </p>

        <div class="shap-words-container">
          ${renderShapTokens(data.shapTokens)}
        </div>

        <div class="shap-legend">
          <div class="shap-legend-item"><div class="shap-dot pos"></div> Credible Support</div>
          <div class="shap-legend-item"><div class="shap-dot neg"></div> Suspicious Marker</div>
        </div>
      </div>

      <!-- Right Column: Claim Decomposition & Evidence RAG -->
      <div class="content-card">
        <div class="card-header-flex">
          <span class="card-title">Claim Decomposition &amp; Sources</span>
          <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">Vector Index</span>
        </div>
        <p style="font-size: 0.84rem; color: #64748B; margin-bottom: 0.85rem;">
          Atomic assertions cross-referenced against vector database records and cited publications.
        </p>

        ${data.claims && data.claims.length > 0 ? data.claims.map((claim, i) => `
          <div class="claim-box">
            <div class="claim-box-header">
              <span class="claim-title-text">Claim 0${i + 1}: "${claim.text}"</span>
              <span class="claim-status-pill ${claim.statusClass}">${claim.status}</span>
            </div>

            ${claim.explanation ? `<div class="claim-reason">${escapeHtml(claim.explanation)}</div>` : ''}

            ${claim.evidence && claim.evidence.length > 0 ? claim.evidence.map(ev => `
              <div class="evidence-quote-box ${ev.usedAsEvidence === false ? 'is-context' : ''}">
                <div class="evidence-source">${ev.sourceId ? `<span class="source-id-chip">${ev.sourceId}</span> ` : ''}${ev.publisher} • <span style="color: #2563EB;">${ev.relationship}</span>${typeof ev.relevance === 'number' ? ` <span class="ev-score">relevance ${ev.relevance.toFixed(2)}</span>` : ''}</div>
                <div class="evidence-text">“${ev.snippet}”</div>
                ${ev.fullTextAvailable === false ? `<div class="ev-note">Full article text was not available through the search API.</div>` : ''}
                ${ev.url && ev.url !== '#' ? `<a href="${ev.url}" target="_blank" rel="noopener noreferrer" class="evidence-link">View Cited Source →</a>` : ''}
              </div>
            `).join('') : `
              <div class="evidence-quote-box">
                <p style="font-size: 0.82rem; color: #64748B;">No direct corroborating records found in index.</p>
              </div>
            `}
          </div>
        `).join('') : `
          <div class="evidence-quote-box">
            <p style="font-size: 0.85rem; color: #64748B;">No atomic claims decomposed.</p>
          </div>
        `}
      </div>

    </div>

    <!-- ===================================================================
         BACKEND EVIDENCE SECTIONS
         Added during backend integration. These render the real, traceable
         data returned by POST /analyze. They appear below the main result and
         replace nothing above.
         =================================================================== -->
    ${renderBackendNotice()}
    ${renderSourceProvenance(data)}
    ${renderUserSubmission(data)}
    ${renderEnsembleAssessment(data)}
    ${renderLimitations(data)}

    <!-- External Fact Checks (if available) -->
    ${data.factChecks && data.factChecks.length > 0 ? `
      <div class="content-card">
        <div class="card-header-flex">
          <span class="card-title">Corroborating Fact-Check Records</span>
          <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">Google Fact Check Tools</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 0.75rem;">
          ${data.factChecks.map(fc => `
            <div class="evidence-quote-box" style="border-left-color: #6366F1;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="font-size: 0.86rem; color: #0F172A;">${fc.publisher}</strong>
                <span style="font-size: 0.75rem; font-weight: 700; color: #6366F1;">${fc.rating}</span>
              </div>
              <p style="font-size: 0.82rem; color: #64748B; margin: 0.35rem 0;">Claim: "${fc.claim}"</p>
              ${fc.url ? `<a href="${fc.url}" target="_blank" rel="noopener noreferrer" class="evidence-link">Read Full Review →</a>` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    <!-- Technical Details Accordion -->
    <div class="content-card">
      <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" id="btn-toggle-tech">
        <span style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">⚙ Technical Model Metadata</span>
        <span style="font-size: 0.82rem; color: #2563EB; font-weight: 600;">${state.showTechDetails ? '▲ Hide' : '▼ View Details'}</span>
      </div>

      ${state.showTechDetails ? `
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color); font-family: monospace; font-size: 0.82rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem;">
          <div><span style="color: #64748B;">Classifier:</span> <strong>${data.technicalDetails?.model || 'DistilBERT'}</strong></div>
          <div><span style="color: #64748B;">Processing Latency:</span> <strong>${data.technicalDetails?.processingTimeMs || 180} ms</strong></div>
          <div><span style="color: #64748B;">Sources Evaluated:</span> <strong>${data.technicalDetails?.ragSourcesChecked || 0} records</strong></div>
          <div><span style="color: #64748B;">Explainer:</span> <strong>${data.explanationMethod}</strong></div>
        </div>
      ` : ''}
    </div>

    <!-- User Feedback -->
    <div class="content-card" style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 0.94rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;">Was this verification assessment accurate and helpful?</div>
      ${state.feedbackSubmitted ? `
        <div style="color: #10B981; font-weight: 600; font-size: 0.9rem;">✓ Thank you! Your feedback helps calibrate system accuracy.</div>
      ` : `
        <div style="display: flex; gap: 0.75rem; justify-content: center;">
          <button class="btn btn-secondary btn-feedback" data-vote="yes" type="button">👍 Yes, Accurate</button>
          <button class="btn btn-secondary btn-feedback" data-vote="no" type="button">👎 Needs Review</button>
        </div>
      `}
    </div>
  `;
}

// ===========================================================================
// BACKEND EVIDENCE SECTIONS
// Added during backend integration. Every value rendered here comes from the
// live POST /analyze response. Missing metadata is shown as
// "Not provided by the source." and is never invented.
// ===========================================================================

const MISSING_TEXT = 'Not provided by the source.';

// A5/A6: the four scores are shown separately so a writing-style percentage can
// never be read as "this claim is X% true".
function renderScoreBreakdown(data) {
  const p = data.provenance;
  if (!p || !p.scores) return '';
  const s = p.scores;
  const e = p.structured || {};

  return `
    <div class="content-card">
      <div class="card-header-flex">
        <span class="card-title">Verification Breakdown</span>
        <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
          ${escapeHtml(s.finalStatus)}
        </span>
      </div>

      <div class="score-grid">
        <div class="score-box">
          <div class="score-label">Writing-Style Model Signal</div>
          <div class="score-value">${s.mlStyleSignal}%</div>
          <div class="score-sub">${escapeHtml(s.mlStyleDirection)}-style</div>
        </div>
        <div class="score-box">
          <div class="score-label">Evidence Relevance</div>
          <div class="score-value">${s.evidenceRelevance}</div>
          <div class="score-sub">best matching source</div>
        </div>
        <div class="score-box">
          <div class="score-label">Source Agreement</div>
          <div class="score-value">${s.sourceAgreementScore}</div>
          <div class="score-sub">${s.independentPublisherCount} independent publisher(s)</div>
        </div>
        <div class="score-box is-primary">
          <div class="score-label">Verification Confidence</div>
          <div class="score-value">${s.verificationConfidence}</div>
          <div class="score-sub">${s.relevantSourceCount} relevant source(s)</div>
        </div>
      </div>

      <div class="notice-inline notice-warn">
        <strong>Writing-Style Model Signal is not factual verification.</strong>
        ${escapeHtml(s.mlDisclaimer)}
      </div>

      ${e.sourceSearch ? `
        <div class="explain-row">
          <div class="explain-label">Source search</div>
          <div class="explain-body">${escapeHtml(e.sourceSearch)}</div>
        </div>` : ''}

      ${e.recommendedNextStep ? `
        <div class="explain-row">
          <div class="explain-label">Recommended next step</div>
          <div class="explain-body">${escapeHtml(e.recommendedNextStep)}</div>
        </div>` : ''}
    </div>
  `;
}

const RELATION_BADGE = {
  SUPPORTS: { icon: '🟢', label: 'Supports the claim' },
  CONTRADICTS: { icon: '🔴', label: 'Contradicts the claim' },
  PARTIALLY_SUPPORTS: { icon: '🟡', label: 'Partially supports the claim' },
  UNRELATED: { icon: '⚪', label: 'Unrelated to the claim' },
  UNKNOWN: { icon: '❔', label: 'Unable to assess' }
};

const EVIDENCE_STATUS_LABEL = {
  RELEVANT: 'Relevant evidence',
  WEAK: 'Weak or partially relevant evidence',
  CONTRADICTORY: 'Contradictory evidence',
  UNRELATED: 'Unrelated result',
  UNKNOWN: 'Unable to assess'
};

const INPUT_TYPE_LABEL = {
  HEADLINE: 'Looks like a headline',
  ARTICLE_CLIP: 'Looks like a paragraph or article clipping',
  FULL_ARTICLE: 'Looks like a full article',
  UNKNOWN: 'Could not determine the shape of this input'
};

function shownValue(value) {
  if (value === null || value === undefined) return MISSING_TEXT;
  const s = String(value).trim();
  return s.length ? escapeHtml(s) : MISSING_TEXT;
}

// Shows the demo-fallback / backend-unreachable warning that the original
// client only logged to the console.
function renderBackendNotice() {
  const err = state.currentResult && state.currentResult.error;
  if (!err) return '';
  return `
    <div class="content-card notice-card notice-error">
      <div class="notice-title">⚠ Backend Status</div>
      <p class="notice-body">${escapeHtml(err)}</p>
    </div>
  `;
}

// Always-visible list of every retrieved link. The expandable cards below still
// hold the full metadata, but URLs must never be hidden behind a click.
function renderSourceLinkList(sources, userUrls) {
  if ((!sources || !sources.length) && (!userUrls || !userUrls.length)) return '';

  return `
    <div class="link-list-block">
      <div class="link-list-title">All links (${(sources || []).length} retrieved${
        userUrls && userUrls.length ? ` + ${userUrls.length} supplied by you` : ''
      })</div>

      ${userUrls && userUrls.length ? userUrls.map(url => `
        <div class="link-row is-user">
          <div class="link-row-head">
            <span class="source-id-chip">USER-001</span>
            <span class="link-row-pub">URL supplied by you</span>
            <span class="link-row-rel">Not opened or verified</span>
          </div>
          <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="link-row-url">${escapeHtml(url)}</a>
        </div>
      `).join('') : ''}

      ${(sources || []).map(s => {
        const badge = RELATION_BADGE[s.claim_relation] || RELATION_BADGE.UNKNOWN;
        const hasUrl = s.url && s.url !== MISSING_TEXT;
        return `
          <div class="link-row ${s.used_in_final_answer ? 'is-used' : ''}">
            <div class="link-row-head">
              <span class="source-id-chip">${shownValue(s.source_id)}</span>
              <span class="link-row-pub">${shownValue(s.publisher)}</span>
              <span class="link-row-rel">${badge.icon} ${badge.label}</span>
              <span class="link-row-score">rel ${Number(s.relevance_score ?? 0).toFixed(2)}</span>
            </div>
            <div class="link-row-title">${shownValue(s.title)}</div>
            ${hasUrl
              ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" class="link-row-url">${escapeHtml(s.url)}</a>`
              : `<div class="link-row-url is-missing">${MISSING_TEXT}</div>`}
            ${s.full_text_available === false
              ? `<div class="ev-note">Full article text was not available through the search API.</div>`
              : ''}
          </div>`;
      }).join('')}
    </div>
  `;
}

function renderSourceProvenance(data) {
  const p = data.provenance;
  if (!p) return '';

  const sources = p.newsApiSources || [];
  const usedCount = sources.filter(s => s.used_in_final_answer).length;
  const queries = p.newsSearch.queries || [];

  let statusBlock = '';
  if (!p.newsSearch.ok) {
    statusBlock = `
      <div class="notice-inline notice-error">
        ${escapeHtml(p.newsSearch.error ||
          'The News API request failed or was rate-limited. Please try again later. This result should be treated as unverified.')}
      </div>`;
  } else if (sources.length === 0) {
    statusBlock = `
      <div class="notice-inline notice-warn">
        No relevant News API sources were found. The result is based only on the
        ML signal and cannot be treated as independently verified.
      </div>`;
  }

  return `
    <div class="content-card">
      <div class="card-header-flex">
        <span class="card-title">Source Provenance and Evidence</span>
        <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
          Agreement: ${shownValue(p.sourceAgreement)}
        </span>
      </div>

      <div class="notice-inline notice-info">
        News API results are related-source evidence. They are not automatically
        proof that a claim is true or false.
      </div>

      ${statusBlock}

      ${renderSourceLinkList(sources, (p.userInput && p.userInput.user_supplied_urls) || [])}

      ${queries.length ? `
        <details class="prov-details">
          <summary>Search queries used (${queries.length})</summary>
          <ul class="query-list">
            ${queries.map(q => `
              <li>
                <code>${shownValue(q.query_id)}</code>
                <strong>${shownValue(q.query_type)}</strong>: ${shownValue(q.query_text)}
              </li>`).join('')}
          </ul>
        </details>` : ''}

      ${sources.length ? `
        <div class="prov-summary-line">
          <strong>${sources.length}</strong> article(s) retrieved —
          <strong>${usedCount}</strong> used as evidence in the final answer.
        </div>

        ${sources.map(s => {
          const badge = RELATION_BADGE[s.claim_relation] || RELATION_BADGE.UNKNOWN;
          const used = s.used_in_final_answer;
          const hasUrl = s.url && s.url !== MISSING_TEXT;
          return `
            <details class="prov-details source-card ${used ? 'is-used' : ''}">
              <summary>
                <span class="source-id-chip">${shownValue(s.source_id)}</span>
                <span class="source-pub">${shownValue(s.publisher)}</span>
                <span class="source-rel">${badge.icon} ${badge.label}</span>
                <span class="source-used ${used ? 'yes' : 'no'}">
                  ${used ? '★ used' : 'not used'}
                </span>
              </summary>

              <div class="source-body">
                <div class="source-headline">${shownValue(s.title)}</div>

                <dl class="source-meta">
                  <dt>Source ID</dt><dd><code>${shownValue(s.source_id)}</code></dd>
                  <dt>Source type</dt><dd><code>${shownValue(s.source_type)}</code></dd>
                  <dt>Publisher</dt><dd>${shownValue(s.publisher)}</dd>
                  <dt>Article URL</dt>
                  <dd>${hasUrl
                    ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" class="evidence-link">${escapeHtml(s.url)}</a>`
                    : MISSING_TEXT}</dd>
                  <dt>Author</dt><dd>${shownValue(s.author)}</dd>
                  <dt>Published at</dt><dd>${shownValue(s.published_at)}</dd>
                  <dt>Description</dt><dd>${shownValue(s.description)}</dd>
                  <dt>Retrieved by query</dt><dd><code>${shownValue(s.retrieval_query)}</code></dd>
                  <dt>Retrieved at</dt><dd>${shownValue(s.retrieved_at)}</dd>
                  <dt>Relevance score</dt><dd>${Number(s.relevance_score ?? 0).toFixed(2)}</dd>
                  <dt>Claim similarity</dt><dd>${Number(s.text_similarity ?? 0).toFixed(2)}</dd>
                  <dt>Full article text available</dt>
                  <dd>${s.full_text_available ? 'Yes' : 'No'}${
                    s.availability_note ? ` — ${escapeHtml(s.availability_note)}` : ''
                  }</dd>
                  <dt>Relation to claim</dt><dd>${badge.icon} ${badge.label}</dd>
                  <dt>Reliability status</dt>
                  <dd>${escapeHtml(EVIDENCE_STATUS_LABEL[s.evidence_status] || 'Unable to assess')}</dd>
                  <dt>Used in final answer</dt><dd>${used ? 'Yes' : 'No'}</dd>
                </dl>

                <div class="source-hint">${shownValue(s.source_quality_hint)}</div>
              </div>
            </details>`;
        }).join('')}
      ` : ''}

      <div class="prov-divider"></div>
      <div class="prov-nonevidence-title">Non-evidence sources used to produce this page</div>
      <div class="prov-nonevidence-grid">
        <div class="nonevidence-box">
          <code>${shownValue(p.modelSource.source_id || 'MODEL-001')}</code>
          <strong>${shownValue(p.modelSource.source_type || 'MODEL_OUTPUT')}</strong>
          <p>${shownValue(p.modelSource.description)}</p>
          <p class="muted">This is not an external source and is never proof.</p>
        </div>
        <div class="nonevidence-box">
          <code>${shownValue(p.aiExplanationSource.source_id || 'AI-001')}</code>
          <strong>${shownValue(p.aiExplanationSource.source_type || 'AI_EXPLANATION')}</strong>
          <p>${shownValue(p.aiExplanationSource.description)}</p>
          <p class="muted">This is an interpretation of the evidence above, not an independent source.</p>
        </div>
      </div>
    </div>
  `;
}

function renderUserSubmission(data) {
  const p = data.provenance;
  if (!p) return '';
  const u = p.userSubmittedSource || {};
  const urls = u.user_supplied_urls || [];

  return `
    <div class="content-card">
      <div class="card-header-flex">
        <span class="card-title">User-Submitted Article or Clip</span>
        <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
          ${shownValue(u.source_id || 'USER-001')}
        </span>
      </div>

      <dl class="source-meta">
        <dt>Source type</dt><dd><code>${shownValue(u.source_type || 'USER_SUBMITTED_TEXT')}</code></dd>
        <dt>Character count</dt><dd>${Number(u.character_count ?? 0)}</dd>
        <dt>Input shape</dt>
        <dd>${escapeHtml(INPUT_TYPE_LABEL[u.input_type] || 'Unknown')}</dd>
      </dl>

      ${urls.length ? `
        <div class="user-url-block">
          <div class="user-url-title">URL supplied by user</div>
          <ul>
            ${urls.map(url => `
              <li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="evidence-link">${escapeHtml(url)}</a></li>
            `).join('')}
          </ul>
          <p class="muted">
            This URL was taken from the text you pasted. It has not been opened
            or checked, and it is not assumed to be genuine or to support the claim.
          </p>
        </div>` : ''}

      <details class="prov-details">
        <summary>Show the exact text that was analysed</summary>
        <pre class="user-text-block">${escapeHtml(u.text || '')}</pre>
        ${u.truncated ? `<p class="muted">This display was truncated for length.</p>` : ''}
      </details>

      <div class="notice-inline notice-warn">
        ${shownValue(u.notice ||
          'This text was supplied by the user. It was not independently verified and may be incomplete or edited.')}
      </div>
    </div>
  `;
}

function renderEnsembleAssessment(data) {
  const p = data.provenance;
  if (!p) return '';
  const ml = p.ml || {};

  if (!ml.available) {
    return `
      <div class="content-card">
        <div class="card-header-flex">
          <span class="card-title">Machine-Learning Ensemble Assessment</span>
        </div>
        <div class="notice-inline notice-warn">
          ${shownValue(ml.note || 'The machine-learning ensemble was not available for this request.')}
        </div>
      </div>`;
  }

  return `
    <div class="content-card">
      <div class="card-header-flex">
        <span class="card-title">Machine-Learning Ensemble Assessment</span>
        <span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
          MODEL-001 • MODEL_OUTPUT
        </span>
      </div>

      <div class="ml-metrics">
        <div class="ml-metric">
          <div class="ml-metric-label">Ensemble prediction</div>
          <div class="ml-metric-value">${shownValue(ml.prediction)}</div>
        </div>
        <div class="ml-metric">
          <div class="ml-metric-label">Ensemble confidence</div>
          <div class="ml-metric-value">${Number(ml.confidence ?? 0)}%</div>
        </div>
        <div class="ml-metric">
          <div class="ml-metric-label">Members agree</div>
          <div class="ml-metric-value">${ml.modelsAgree ? 'Yes' : 'No'}</div>
        </div>
      </div>

      ${(ml.votes || []).length ? `
        <table class="votes-table">
          <thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th></tr></thead>
          <tbody>
            ${ml.votes.map(v => `
              <tr>
                <td>${shownValue(v.model_name)}</td>
                <td>${shownValue(v.prediction)}</td>
                <td>${Number(v.confidence ?? 0)}%</td>
              </tr>`).join('')}
          </tbody>
        </table>` : ''}

      ${ml.note ? `<div class="notice-inline notice-warn">${escapeHtml(ml.note)}</div>` : ''}

      <p class="muted">${shownValue(ml.interpretation)}</p>
    </div>
  `;
}

function renderLimitations(data) {
  const p = data.provenance;
  if (!p) return '';
  const limits = p.limitations || [];
  const warnings = p.systemWarnings || [];

  return `
    <div class="content-card">
      <div class="card-header-flex">
        <span class="card-title">Recommended Next Step &amp; Limitations</span>
        ${p.generatedBy ? `<span style="font-size: 0.75rem; color: #64748B; font-weight: 600;">
          Explanation: ${shownValue(p.generatedBy)}
        </span>` : ''}
      </div>

      ${p.recommendedAction ? `
        <div class="notice-inline notice-info">${escapeHtml(p.recommendedAction)}</div>` : ''}

      ${limits.length ? `
        <div class="limits-title">Limitations of this analysis</div>
        <ul class="limits-list">
          ${limits.map(l => `<li>${escapeHtml(l)}</li>`).join('')}
        </ul>` : ''}

      ${warnings.length ? `
        <details class="prov-details">
          <summary>System warnings (${warnings.length})</summary>
          <ul class="limits-list">
            ${warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}
          </ul>
        </details>` : ''}

      ${p.requestId ? `
        <p class="muted">Request ID: ${shownValue(p.requestId)} • Analysed at: ${shownValue(p.analyzedAt)}</p>` : ''}
    </div>
  `;
}

function renderFooter() {
  return `
    <footer class="footer">
      <div style="max-width: 960px; margin: 0 auto; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem;">
        <div>
          <strong style="color: #0F172A; font-size: 0.95rem;">TruthLens AI</strong>
          <p style="font-size: 0.8rem; margin-top: 0.2rem;">“See Beyond the Headlines. Discover the Truth.”</p>
        </div>
        <div style="font-size: 0.8rem;">
          Responsible AI • Transparent Attribution • Evidence-Grounded Verification
        </div>
      </div>
    </footer>
  `;
}

function renderShapTokens(tokens) {
  if (!tokens || tokens.length === 0) {
    return `<span style="color: #64748B;">Attribution tokens unavailable for this submission.</span>`;
  }

  return tokens.map(t => {
    let wordClass = "";
    if (t.weight > 0.15) wordClass = "shap-pos";
    else if (t.weight < -0.15) wordClass = "shap-neg";

    return `<span class="shap-word ${wordClass}" title="Attribution: ${t.weight > 0 ? '+' : ''}${t.weight.toFixed(2)}">${escapeHtml(t.word)}</span>`;
  }).join(' ');
}

function getVerdictIcon(label) {
  if (label.includes('Supported by Retrieved Evidence')) return '✓';
  if (label.includes('Contradicted')) return '✕';
  if (label.includes('Partially')) return '◑';
  if (label.includes('Unable to Verify')) return '⊘';
  if (label.includes('Needs Verification')) return '⚠';
  if (label.includes('Real')) return '✓';
  if (label.includes('Fake')) return '✕';
  if (label.includes('Suspicious')) return '⚠';
  return '◌';
}

function bindEvents() {
  // Splash slide click anywhere to slide right
  const splashSlide = document.getElementById('slide-splash');
  if (splashSlide) {
    splashSlide.addEventListener('click', () => {
      goToSlide(1);
    });
  }

  // Navigation buttons
  document.querySelectorAll('.btn-nav-back').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      goToSlide(1);
    });
  });

  document.querySelectorAll('[data-action="go-splash"], .btn-nav-splash').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      goToSlide(0);
    });
  });

  // Textarea input
  const textarea = document.querySelector('#news-input');
  const btnAnalyze = document.querySelector('#btn-analyze');
  const btnClear = document.querySelector('#btn-clear');
  const btnPaste = document.querySelector('#btn-paste');
  const charCount = document.querySelector('#char-count');

  if (textarea) {
    textarea.addEventListener('input', (e) => {
      state.inputText = e.target.value;
      state.validationError = '';
      if (charCount) charCount.textContent = state.inputText.length;
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', (e) => {
      e.stopPropagation();
      state.inputText = '';
      state.validationError = '';
      if (textarea) textarea.value = '';
      if (charCount) charCount.textContent = '0';
    });
  }

  if (btnPaste) {
    btnPaste.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          state.inputText = text;
          state.validationError = '';
          if (textarea) textarea.value = text;
          if (charCount) charCount.textContent = text.length;
        }
      } catch (err) {
        console.warn("Clipboard read restricted", err);
      }
    });
  }

  // Format Tabs
  document.querySelectorAll('.format-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.stopPropagation();
      state.inputType = tab.getAttribute('data-type');
      document.querySelectorAll('.format-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
    });
  });

  // Evaluate & Slide Right to Result Report
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', async (e) => {
      e.stopPropagation();
      const trimmed = state.inputText.trim();
      if (trimmed.length === 0) {
        state.validationError = 'Please enter an article, headline, or claim to verify.';
        renderApp();
        return;
      }
      if (trimmed.length < 15) {
        state.validationError = 'Please provide sufficient context for meaningful verification (at least 15 characters).';
        renderApp();
        return;
      }

      state.isLoading = true;
      state.validationError = '';
      state.feedbackSubmitted = false;

      // Slide right to Result Report
      goToSlide(2);
      const resultContainer = document.getElementById('result-content-container');
      if (resultContainer) {
        resultContainer.innerHTML = renderResultDashboard();
      }

      const result = await verifyArticle(state.inputText);
      state.isLoading = false;
      state.currentResult = result;

      if (resultContainer) {
        resultContainer.innerHTML = renderResultDashboard();
        bindResultEvents();
      }
    });
  }

  // Benchmark Presets -> Slide Right to Result Report
  document.querySelectorAll('.preset-card').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const sampleId = btn.getAttribute('data-sample');
      const sample = DEMO_SAMPLES.find(s => s.id === sampleId);
      if (sample) {
        state.inputText = sample.rawText;
        state.isLoading = true;
        state.validationError = '';
        state.feedbackSubmitted = false;

        goToSlide(2);
        const resultContainer = document.getElementById('result-content-container');
        if (resultContainer) {
          resultContainer.innerHTML = renderResultDashboard();
        }

        const result = await verifyArticle(sample.rawText, sampleId);
        state.isLoading = false;
        state.currentResult = result;

        if (resultContainer) {
          resultContainer.innerHTML = renderResultDashboard();
          bindResultEvents();
        }
      }
    });
  });

  bindResultEvents();
}

function bindResultEvents() {
  document.querySelectorAll('.btn-nav-back').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      goToSlide(1);
    });
  });

  const btnTech = document.querySelector('#btn-toggle-tech');
  if (btnTech) {
    btnTech.addEventListener('click', (e) => {
      e.stopPropagation();
      state.showTechDetails = !state.showTechDetails;
      const resultContainer = document.getElementById('result-content-container');
      if (resultContainer) {
        resultContainer.innerHTML = renderResultDashboard();
        bindResultEvents();
      }
    });
  }

  document.querySelectorAll('.btn-feedback').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const vote = btn.getAttribute('data-vote');
      await submitFeedback({
        article_id: state.currentResult?.data?.id || "demo",
        helpful: vote === 'yes',
        user_verdict: state.currentResult?.data?.verdictLabel || "Unsure"
      });
      state.feedbackSubmitted = true;
      const resultContainer = document.getElementById('result-content-container');
      if (resultContainer) {
        resultContainer.innerHTML = renderResultDashboard();
        bindResultEvents();
      }
    });
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Start
init();
