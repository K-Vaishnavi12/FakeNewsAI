import { useState } from 'react'
import './App.css'

const PRESET_EXAMPLES = [
  {
    title: 'Verified Real News',
    badge: 'real',
    text: 'James Webb Space Telescope captures unprecedented deep-space observations of ancient galaxy clusters, NASA scientists confirmed on Friday.',
  },
  {
    title: 'Mars Water Debunk',
    badge: 'debunk',
    text: 'Scientists discovered drinkable water on Mars this week.',
  },
  {
    title: 'Fabricated Conspiracy',
    badge: 'fake',
    text: 'BREAKING: Leaked shocking video exposes secret billionaire deep-state cabal plotting global economic collapse inside hidden underground base!',
  },
  {
    title: 'Unverified Claim',
    badge: 'neutral',
    text: 'A secret laboratory has reportedly synthesized a miracle compound that completely reverses human biological aging in three weeks.',
  }
]

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    if (e) e.preventDefault()
    if (!text.trim()) return

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
  const evidenceBreakdown = finalAnalysis?.evidence_breakdown || {}
  const fakePercent = Math.round((mlData?.fake_probability || 0) * 100)
  const realPercent = Math.round((mlData?.real_probability || 0) * 100)
  const overallScore = Math.round((finalAnalysis?.confidence_score || 0) * 100)

  const verdictType = (finalAnalysis?.verdict_type || '').toLowerCase()
  const verdictClass = verdictType.includes('real')
    ? 'verdict-real'
    : verdictType.includes('fake')
    ? 'verdict-fake'
    : verdictType.includes('disputed')
    ? 'verdict-disputed'
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
            <span>AI AGENT FACT-CHECKER & EVIDENCE CLASSIFIER</span>
          </div>
          <h1>Fake News Intelligence</h1>
          <p className="subtitle">
            Multi-branch verification combining a <strong>44.9k-trained ML Classifier</strong>, <strong>Multi-Query Fact-Checking</strong>, and <strong>Semantic Evidence Classification</strong>.
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
              <span>ML Model</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="step-num">3</span>
              <span>Multi-Query Search</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step highlight">
              <span className="step-num">4</span>
              <span>Evidence Classification</span>
            </div>
            <div className="flow-arrow">→</div>
            <div className="flow-step">
              <span className="step-num">5</span>
              <span>Decision Matrix</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          {/* Input Section */}
          <section className="card input-card">
            <div className="card-header">
              <h2>Verify Article or Claim</h2>
              <span className="char-count">{text.length} characters</span>
            </div>

            {/* Quick Presets */}
            <div className="presets-bar">
              <span className="presets-label">Try scenario:</span>
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
                placeholder="Paste news headline, article body, or controversial claim here to run deep multi-branch analysis..."
                rows={5}
                required
              />

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
                  disabled={loading || !text.trim()}
                >
                  {loading ? (
                    <>
                      <span className="spinner" />
                      <span>Classifying Evidence Streams...</span>
                    </>
                  ) : (
                    <>
                      <span>Run Verification Agent</span>
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
              <h3>Analyzing Information Streams & Classifying Evidence...</h3>
              <div className="loading-steps">
                <div className="loading-step active">
                  <span className="check-icon">✓</span>
                  <span>Branch 1: Computing TF-IDF & Stylometric probabilities</span>
                </div>
                <div className="loading-step active">
                  <span className="check-icon">✓</span>
                  <span>Branch 2: Searching live news & fact-check databases (multi-query)</span>
                </div>
                <div className="loading-step active">
                  <span className="spinner-mini" />
                  <span>Branch 3: Classifying article stances (Supports, Contradicts, Neutral)</span>
                </div>
                <div className="loading-step active">
                  <span className="spinner-mini" />
                  <span>Convergence: Fusing Decision Matrix into definitive final verdict</span>
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
                    <span className="verdict-tag">FINAL VERDICT & SYNTHESIS</span>
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

                {/* Evidence Stance Breakdown Bar */}
                {evidenceBreakdown && (
                  <div className="evidence-summary-bar">
                    <span className="evidence-summary-title">Evidence Stance Breakdown:</span>
                    <div className="evidence-pills">
                      <span className="evidence-pill pill-support">
                        ✓ {evidenceBreakdown.supports_count || 0} Supports Claim
                      </span>
                      <span className="evidence-pill pill-contradict">
                        ✗ {evidenceBreakdown.contradicts_count || 0} Contradicts / Debunked
                      </span>
                      <span className="evidence-pill pill-neutral">
                        ~ {evidenceBreakdown.neutral_count || 0} Neutral / Related
                      </span>
                    </div>
                  </div>
                )}

                {/* Synthesis Highlights */}
                <div className="synthesis-grid">
                  {finalAnalysis?.ml_insights && (
                    <div className="insight-box">
                      <h4>📊 Branch 1: Statistical & Linguistic Analysis</h4>
                      <p>{finalAnalysis.ml_insights}</p>
                    </div>
                  )}

                  {finalAnalysis?.news_cross_check && (
                    <div className="insight-box">
                      <h4>🌐 Branch 2 & 3: Evidence Cross-Check</h4>
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
                    <strong>💡 Agent Recommendation: </strong>
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
                      <span className="card-sub">Branch 1: TF-IDF + Logistic Regression</span>
                    </div>
                    <span className="accuracy-pill">98.8% Accuracy</span>
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
                      <h4>Influential Linguistic Cues</h4>
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
                    <span>Model trained on 44,898 verified historical articles</span>
                  </div>
                </section>

                {/* Column 2: Classified Live News Evidence */}
                <section className="card news-card">
                  <div className="card-header">
                    <div>
                      <h3>Classified Evidence</h3>
                      <span className="card-sub">Branch 2 & 3: Multi-Query Live News & Stance Analysis</span>
                    </div>
                    <span className="articles-count-badge">
                      {newsSources.length} Source{newsSources.length !== 1 ? 's' : ''} Analyzed
                    </span>
                  </div>

                  <div className="articles-scroll">
                    {newsSources.length > 0 ? (
                      newsSources.map((art, idx) => {
                        const stanceType = (art.stance_type || 'neutral').toLowerCase()
                        const stanceBadgeClass = stanceType === 'support'
                          ? 'stance-badge-support'
                          : stanceType === 'contradict'
                          ? 'stance-badge-contradict'
                          : 'stance-badge-neutral'

                        return (
                          <div key={idx} className={`article-item article-${stanceType}`}>
                            <div className="article-header">
                              <span className="article-source">{art.source || 'News Source'}</span>
                              <span className={`stance-badge ${stanceBadgeClass}`}>
                                {art.stance || 'NEUTRAL / RELATED'}
                              </span>
                            </div>

                            <h4 className="article-title">{art.title}</h4>

                            {art.stance_reason && (
                              <div className="stance-reason-box">
                                <span className="stance-reason-icon">🔍</span>
                                <span>{art.stance_reason}</span>
                              </div>
                            )}

                            {art.description && (
                              <p className="article-snippet">{art.description}</p>
                            )}

                            <div className="article-footer">
                              {art.published_at && (
                                <span className="article-date">
                                  {new Date(art.published_at).toLocaleDateString(undefined, {
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric'
                                  })}
                                </span>
                              )}
                              {art.url && (
                                <a
                                  href={art.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="article-link-btn"
                                >
                                  <span>Read Source</span>
                                  <span>↗</span>
                                </a>
                              )}
                            </div>
                          </div>
                        )
                      })
                    ) : (
                      <div className="no-articles-msg">
                        <p>No directly matching articles were found across live news & fact-check streams for this query.</p>
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
