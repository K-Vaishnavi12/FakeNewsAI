import { useEffect, useState } from 'react'
import './App.css'

// Keep in sync with the backend's MAX_INPUT_CHARS. The server is the real
// authority; this only gives the user immediate feedback instead of a 400.
const MAX_INPUT_CHARS = 10000

// Only these schemes may appear in an href we render.
const SAFE_URL_SCHEMES = ['http:', 'https:']

/**
 * Return `url` only if it is a syntactically valid http(s) URL, else null.
 *
 * Article URLs come from NewsAPI / Google News RSS, i.e. from outside our
 * trust boundary. React escapes text nodes, but it does NOT sanitise the
 * `href` attribute: rendering `href="javascript:alert(document.cookie)"`
 * produces a working XSS payload that fires on click. Every untrusted URL
 * must therefore pass through this function before being rendered.
 *
 * @param {unknown} url Candidate URL from the backend.
 * @returns {string|null} The safe URL, or null if it must not be linked.
 */
function safeUrl(url) {
  if (typeof url !== 'string' || !url.trim()) return null
  try {
    const parsed = new URL(url, window.location.origin)
    return SAFE_URL_SCHEMES.includes(parsed.protocol) ? parsed.href : null
  } catch {
    // Not parseable as a URL at all -> never render it as a link.
    return null
  }
}

const PRESET_EXAMPLES = [
  {
    title: 'Real News',
    badge: 'real',
    text: 'James Webb Space Telescope captures unprecedented deep-space observations of ancient galaxy clusters, NASA scientists confirmed on Friday.',
  },
  {
    title: 'Disinformation',
    badge: 'fake',
    text: 'BREAKING: Leaked shocking video exposes secret billionaire deep-state cabal plotting global economic collapse inside hidden underground base!',
  },
  {
    title: 'Ambiguous Claim',
    badge: 'neutral',
    text: 'A secret laboratory has reportedly synthesized a miracle compound that completely reverses human biological aging in three weeks.',
  }
]

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  // Model metadata (accuracy, type) is read from the backend rather than
  // hardcoded in the UI, so it can never drift from the deployed model.
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/health', { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => { if (data) setModelInfo(data) })
      .catch(() => { /* health is informational only; ignore failures */ })
    return () => controller.abort()
  }, [])

  const isOverLimit = text.length > MAX_INPUT_CHARS

  async function handleSubmit(e) {
    if (e) e.preventDefault()
    if (!text.trim() || isOverLimit) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), page_size: 6 })
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || errorData.error || `Server error: ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to complete analysis')
    } finally {
      setLoading(false)
    }
  }

  function handlePresetClick(sampleText) {
    setText(sampleText)
  }

  const finalAnalysis = result?.final_analysis
  const mlData = result?.ml_classifier
  const newsSources = result?.news_sources || []
  const fakePercent = Math.round((mlData?.fake_probability || 0) * 100)
  const realPercent = Math.round((mlData?.real_probability || 0) * 100)
  const overallScore = Math.round((finalAnalysis?.confidence_score || 0) * 100)

  const verdictType = (finalAnalysis?.verdict_type || '').toLowerCase()
  const verdictClass = verdictType.includes('real')
    ? 'verdict-real'
    : verdictType.includes('fake')
    ? 'verdict-fake'
    : 'verdict-unverified'

  return (
    <div className="app-root">
      {/* Background ambient glow */}
      <div className="ambient-glow glow-1" />
      <div className="ambient-glow glow-2" />

      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="logo-badge">
            <span className="logo-dot" />
            <span>AI AGENT FACT-CHECKER</span>
          </div>
          <h1>Fake News Detection</h1>
          <p className="subtitle">
            Dual-branch verification combining a <strong>TF-IDF ML classifier</strong>,{' '}
            <strong>live news corroboration</strong>, and a{' '}
            <strong>deterministic synthesis engine</strong>.
          </p>

          {/* Architecture Pipeline Flow Banner */}
          <div className="pipeline-flow">
            <div className="flow-step">
              <span className="step-num">1</span>
              <span>Input Claim</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="step-num">2</span>
              <span>AI Agent</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step highlight">
              <span className="step-num">3</span>
              <span>ML Model + NewsAPI</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="step-num">4</span>
              <span>Synthesis</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          {/* Input Section */}
          <section className="card input-card">
            <div className="card-header">
              <h2>Verify Article or Claim</h2>
              <span className="char-count">
                {text.length.toLocaleString()} / {MAX_INPUT_CHARS.toLocaleString()} characters
              </span>
            </div>

            {/* Quick Presets */}
            <div className="presets-bar">
              <span className="presets-label">Try example:</span>
              <div className="presets-buttons">
                {PRESET_EXAMPLES.map((ex, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className={`preset-btn preset-${ex.badge}`}
                    onClick={() => handlePresetClick(ex.text)}
                  >
                    {ex.title}
                  </button>
                ))}
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <textarea
                id="claim-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste news headline, article body, or controversial claim here to run deep analysis..."
                rows={5}
                maxLength={MAX_INPUT_CHARS}
                required
              />

              {isOverLimit && (
                <p className="input-warning">
                  Input exceeds the {MAX_INPUT_CHARS.toLocaleString()}-character limit.
                </p>
              )}

              <div className="form-actions">
                {text && (
                  <button
                    type="button"
                    className="clear-btn"
                    onClick={() => { setText(''); setResult(null); setError(null); }}
                  >
                    Clear
                  </button>
                )}
                <button
                  type="submit"
                  className="analyze-btn"
                  disabled={loading || !text.trim() || isOverLimit}
                >
                  {loading ? (
                    <>
                      <span className="spinner" />
                      <span>Running Detection Pipeline...</span>
                    </>
                  ) : (
                    <>
                      <span>Run Verification</span>
                      <span className="btn-icon">⚡</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </section>

          {/* Loading Indicator Steps */}
          {loading && (
            <div className="loading-card card">
              <div className="loading-spinner-ring" />
              <h3>Analyzing Information Streams...</h3>
              <div className="loading-steps">
                <div className="loading-step active">
                  <span className="check-icon">✓</span>
                  <span>Branch A: Computing TF-IDF & Logistic Regression probabilities</span>
                </div>
                <div className="loading-step active">
                  <span className="check-icon">✓</span>
                  <span>Branch B: Searching live verified news databases</span>
                </div>
                <div className="loading-step active">
                  <span className="spinner-mini" />
                  <span>Convergence: Scoring corroboration and generating the final synthesis</span>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-card card">
              <div className="error-icon">⚠️</div>
              <div className="error-text">
                <h3>Analysis Failed</h3>
                <p>{error}</p>
              </div>
            </div>
          )}

          {/* Analysis Results Dashboard */}
          {result && !loading && (
            <div className="results-container">
              {/* Top Hero: Final Analysis Card */}
              <section className={`card verdict-card ${verdictClass}`}>
                <div className="verdict-header">
                  <div className="verdict-badge-group">
                    <span className="verdict-tag">FINAL VERDICT</span>
                    <h2 className="verdict-title">{finalAnalysis?.verdict || 'Analysis Complete'}</h2>
                  </div>

                  <div className="confidence-meter">
                    <div className="confidence-label">Agent Confidence</div>
                    <div className="confidence-value">{overallScore}%</div>
                  </div>
                </div>

                <div className="executive-summary">
                  <p>{finalAnalysis?.executive_summary}</p>
                </div>

                {/* Synthesis Highlights */}
                <div className="synthesis-grid">
                  {finalAnalysis?.ml_insights && (
                    <div className="insight-box">
                      <h4>📊 Statistical Findings</h4>
                      <p>{finalAnalysis.ml_insights}</p>
                    </div>
                  )}

                  {finalAnalysis?.news_cross_check && (
                    <div className="insight-box">
                      <h4>🌐 News Coverage Cross-Check</h4>
                      <p>{finalAnalysis.news_cross_check}</p>
                    </div>
                  )}
                </div>

                {/* Red Flags & Recommendations */}
                {finalAnalysis?.red_flags && finalAnalysis.red_flags.length > 0 && (
                  <div className="red-flags-section">
                    <h4>🚩 Risk Factors / Red Flags:</h4>
                    <ul>
                      {finalAnalysis.red_flags.map((flag, i) => (
                        <li key={i}>{flag}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {finalAnalysis?.recommendations && (
                  <div className="recommendation-box">
                    <strong>💡 Recommendation: </strong>
                    <span>{finalAnalysis.recommendations}</span>
                  </div>
                )}
              </section>

              {/* Dual-Column Intelligence Breakdown */}
              <div className="two-col-grid">
                {/* Column 1: ML Statistical Model Output */}
                <section className="card ml-card">
                  <div className="card-header">
                    <div>
                      <h3>ML Classifier</h3>
                      <span className="card-sub">{mlData?.model_type || 'TF-IDF + Logistic Regression'}</span>
                    </div>
                    {/* Accuracy comes from the trained model's saved metadata
                        via /api/health, not a hardcoded string. */}
                    <span className="accuracy-pill">
                      {mlData?.model_accuracy || modelInfo?.model_accuracy || 'Accuracy unknown'}
                      {(mlData?.model_accuracy || modelInfo?.model_accuracy) ? ' Accuracy' : ''}
                    </span>
                  </div>

                  {/* Probability Gauges */}
                  <div className="probability-section">
                    <div className="prob-bar-container">
                      <div className="prob-label-row">
                        <span className="prob-name fake-color">Fake News Probability</span>
                        <span className="prob-value fake-color">{fakePercent}%</span>
                      </div>
                      <div className="bar-track">
                        <div
                          className="bar-fill fake-bar"
                          style={{ width: `${fakePercent}%` }}
                        />
                      </div>
                    </div>

                    <div className="prob-bar-container">
                      <div className="prob-label-row">
                        <span className="prob-name real-color">Real News Probability</span>
                        <span className="prob-value real-color">{realPercent}%</span>
                      </div>
                      <div className="bar-track">
                        <div
                          className="bar-fill real-bar"
                          style={{ width: `${realPercent}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Salient Linguistic Tokens */}
                  {mlData?.top_signals && mlData.top_signals.length > 0 && (
                    <div className="signals-section">
                      <h4>Influential Word Cues</h4>
                      <div className="signals-chips">
                        {mlData.top_signals.map((sig, i) => (
                          <span
                            key={i}
                            className={`signal-chip signal-${sig.impact}`}
                            title={`Impact weight: ${sig.weight}`}
                          >
                            <span className="sig-word">{sig.word}</span>
                            <span className="sig-badge">{sig.impact}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="ml-footnote">
                    <span>
                      {modelInfo?.model_trained_at
                        ? `Model trained ${modelInfo.model_trained_at}`
                        : 'Model metadata unavailable'}
                    </span>
                  </div>
                </section>

                {/* Column 2: Live Verified News Sources */}
                <section className="card news-card">
                  <div className="card-header">
                    <div>
                      <h3>Live News Verification</h3>
                      <span className="card-sub">NewsAPI & Real-time Feeds</span>
                    </div>
                    <span className="articles-count-badge">
                      {newsSources.length} Source{newsSources.length !== 1 ? 's' : ''} Found
                    </span>
                  </div>

                  <div className="articles-scroll">
                    {newsSources.length > 0 ? (
                      newsSources.map((art, idx) => {
                        // Sanitise before rendering: a malicious search result
                        // could otherwise supply a javascript: URL.
                        const href = safeUrl(art.url)
                        return (
                          <div key={idx} className="article-item">
                            <div className="article-header">
                              <span className="article-source">{art.source || 'News Source'}</span>
                              {art.published_at && (
                                <span className="article-date">
                                  {new Date(art.published_at).toLocaleDateString(undefined, {
                                    month: 'short',
                                    day: 'numeric'
                                  })}
                                </span>
                              )}
                            </div>
                            <h4 className="article-title">{art.title}</h4>
                            {art.description && (
                              <p className="article-snippet">{art.description}</p>
                            )}
                            {href && (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer nofollow"
                                className="article-link-btn"
                              >
                                <span>Read Source</span>
                                <span>↗</span>
                              </a>
                            )}
                          </div>
                        )
                      })
                    ) : (
                      <div className="no-articles-msg">
                        <p>No directly matching articles were found in live news streams for this exact query.</p>
                      </div>
                    )}
                  </div>
                </section>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
