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
            
            <div class="splash-logo-circle">
              <span>🔍</span>
            </div>

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

            <div class="splash-features-bar">
              <div class="splash-feature-item">✓ Fact &amp; Claim Inspection</div>
              <div class="splash-feature-item">✓ SHAP Linguistic Attribution</div>
              <div class="splash-feature-item">✓ ChromaDB Source Retrieval</div>
              <div class="splash-feature-item">✓ Calibrated Trust Scoring</div>
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

      <div class="gauge-box" title="Calibrated Trust Score (0 - 100 scale)">
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle class="gauge-bg" cx="60" cy="60" r="55" />
          <circle class="gauge-fill ${verdictClass}" cx="60" cy="60" r="55" 
            stroke-dasharray="345.57" 
            stroke-dashoffset="${strokeDashoffset}" 
            transform="rotate(-90 60 60)" />
        </svg>
        <div class="gauge-center-text">
          <div class="gauge-score-number">${score}</div>
          <div class="gauge-score-label">Trust Score</div>
        </div>
      </div>
    </div>

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

        <div class="claims-list-container">
          ${data.claims && data.claims.length > 0 ? data.claims.map((claim, i) => `
            <div class="claim-box">
              <div class="claim-box-header">
                <span class="claim-title-text">Claim 0${i + 1}: "${claim.text}"</span>
                <span class="claim-status-pill ${claim.statusClass}">${claim.status}</span>
              </div>

              ${claim.evidence && claim.evidence.length > 0 ? claim.evidence.map(ev => `
                <div class="evidence-quote-box">
                  <div class="evidence-source">${ev.publisher} • <span style="color: #2563EB;">${ev.relationship} Evidence</span></div>
                  <div class="evidence-text">“${ev.snippet}”</div>
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

    </div>

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
  if (label.includes('Real') || label.includes('Likely Real')) return '✓';
  if (label.includes('Fake') || label.includes('Likely Fake')) return '✕';
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
