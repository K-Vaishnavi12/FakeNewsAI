import { useCallback, useEffect, useState } from 'react'
import LaunchScreen from './components/LaunchScreen.jsx'
import './App.css'

const MAX_INPUT_CHARS = 10000

const SAFE_URL_SCHEMES = ['http:', 'https:']

function safeUrl(url) {
  if (typeof url !== 'string' || !url.trim()) return null
  try {
    const parsed = new URL(url, window.location.origin)
    return SAFE_URL_SCHEMES.includes(parsed.protocol) ? parsed.href : null
  } catch {
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
  },
]

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [entered, setEntered] = useState(false)
  // 'input' | 'results'
  const [page, setPage] = useState('input')

  const handleEnter = useCallback(() => setEntered(true), [])

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
    setPage('results')

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), page_size: 6 }),
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

  function handleBackToInput() {
    setPage('input')
    setResult(null)
    setError(null)
  }

  const finalAnalysis = result?.final_analysis
  const mlData = result?.ml_classifier
  const newsSources = result?.news_sources || []
  const fakePercent = Math.round((mlData?.fake_probability || 0) * 100)
  const realPercent = Math.round((mlData?.real_probability || 0) * 100)
  const overallScore = Math.round((finalAnalysis?.confidence_score || 0) * 100)

  const verdictType = (finalAnalysis?.verdict_type || '').toLowerCase()
  const verdictClass = verdictType.includes('real')
    ? 'real'
    : verdictType.includes('fake')
    ? 'fake'
    : 'unverified'

  const todayStr = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  return (
    <>
      {!entered && <LaunchScreen onEnter={handleEnter} />}

      {/* ══════════════════════════════════════════════
          PAGE 2 — Dark Input Page
          ══════════════════════════════════════════════ */}
      <div className={`page-input ${page === 'input' ? 'page--active' : 'page--exit'}`}>
        {/* Dark newspaper texture layers */}
        <div className="page-input__bg" />
        <div className="page-input__overlay" />
        <div className="page-input__vignette" />

        <div className="page-input__content">
          {/* Masthead */}
          <header className="dark-header">
            <div className="dark-badge">
              <span className="dark-badge__dot" />
              <span>AI Agent Fact-Checker</span>
            </div>

            <h1 className="dark-title">Fake News Detection</h1>

            <div className="dark-divider">
              <span className="dark-divider__dot" />
            </div>

            <p className="dark-subtitle">
              Dual-branch verification combining a <strong>TF-IDF ML classifier</strong>,{' '}
              <strong>live news corroboration</strong>, and a{' '}
              <strong>deterministic synthesis engine</strong>.
            </p>

            {/* Pipeline Strip */}
            <div className="dark-pipeline">
              <span className="dp-step">
                <span className="dp-num">1</span> Input Claim
              </span>
              <span className="dp-arrow">→</span>
              <span className="dp-step">
                <span className="dp-num">2</span> AI Agent
              </span>
              <span className="dp-arrow">→</span>
              <span className="dp-step dp-step--active">
                <span className="dp-num dp-num--active">3</span> ML Model + NewsAPI
              </span>
              <span className="dp-arrow">→</span>
              <span className="dp-step">
                <span className="dp-num">4</span> Synthesis
              </span>
            </div>
          </header>

          {/* Input Card */}
          <section className="dark-card">
            <div className="dark-card__header">
              <h2 className="dark-card__title">Submit Article or Claim for Verification</h2>
              <span className="dark-card__count">
                {text.length.toLocaleString()} / {MAX_INPUT_CHARS.toLocaleString()}
              </span>
            </div>

            <div className="dark-presets">
              <span className="dark-presets__label">Try example:</span>
              {PRESET_EXAMPLES.map((ex, idx) => (
                <button
                  key={idx}
                  type="button"
                  className={`dark-preset-btn dark-preset-btn--${ex.badge}`}
                  onClick={() => handlePresetClick(ex.text)}
                >
                  {ex.title}
                </button>
              ))}
            </div>

            <form onSubmit={handleSubmit}>
              <textarea
                id="claim-input"
                className="dark-textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste news headline, article body, or controversial claim here to run deep analysis..."
                rows={6}
                maxLength={MAX_INPUT_CHARS}
                required
              />

              {isOverLimit && (
                <p className="input-warning">
                  Input exceeds the {MAX_INPUT_CHARS.toLocaleString()}-character limit.
                </p>
              )}

              <div className="dark-actions">
                {text && (
                  <button
                    type="button"
                    className="dark-clear-btn"
                    onClick={() => { setText(''); setResult(null); setError(null) }}
                  >
                    Clear
                  </button>
                )}
                <button
                  type="submit"
                  id="run-verification-btn"
                  className="dark-analyze-btn"
                  disabled={!text.trim() || isOverLimit}
                >
                  <span>Run Verification</span>
                  <span className="btn-icon">⚡</span>
                </button>
              </div>
            </form>
          </section>

          <p className="dark-footer">Click anywhere to begin · Powered by dual-branch AI analysis</p>
        </div>
      </div>

      {/* ══════════════════════════════════════════════
          PAGE 3 — Newspaper Results Page
          ══════════════════════════════════════════════ */}
      <div className={`page-results ${page === 'results' ? 'page--active' : 'page--hidden'}`}>
        <div className="page-results__bg" />
        <div className="page-results__overlay" />

        <div className="results-page-container">

          {/* ── Newspaper Masthead ── */}
          <div className="results-topbar">
            <button className="back-btn" onClick={handleBackToInput} id="back-to-input-btn">
              ← New Analysis
            </button>

            <div className="results-topbar__meta">
              <span>AI Fact-Check Report</span>
              <span>Verified by Dual-Branch AI</span>
              <span>Est. 2024</span>
            </div>

            <div className="results-topbar__logo">Fake News Detection</div>

            <div className="results-topbar__tagline">
              "Truth Through Technology — AI-Powered Fact Verification System"
            </div>

            <div className="results-topbar__date">{todayStr}</div>
          </div>

          <div className="results-rule" />

          <main className="results-main">

            {/* ── Loading ── */}
            {loading && (
              <div className="loading-card">
                <div className="loading-spinner-ring" />
                <h3>Setting the Presses… Analyzing Information Streams</h3>
                <div className="loading-steps">
                  <div className="loading-step">
                    <span className="check-icon">✓</span>
                    <span>Branch A: Computing TF-IDF &amp; Logistic Regression probabilities</span>
                  </div>
                  <div className="loading-step">
                    <span className="check-icon">✓</span>
                    <span>Branch B: Searching live verified news databases</span>
                  </div>
                  <div className="loading-step">
                    <span className="spinner-mini" />
                    <span>Convergence: Scoring corroboration and generating final synthesis</span>
                  </div>
                </div>
              </div>
            )}

            {/* ── Error ── */}
            {error && (
              <div className="error-card">
                <div className="error-icon">⚠️</div>
                <div className="error-text">
                  <h3>Analysis Failed</h3>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {/* ── Newspaper Results ── */}
            {result && !loading && (
              <div className="results-container">

                {/* FRONT-PAGE VERDICT BANNER */}
                <div className="verdict-banner">
                  <div className="verdict-kicker">Final Verdict · AI Analysis Report</div>

                  <h2 className={`verdict-headline verdict-headline--${verdictClass}`}>
                    {finalAnalysis?.verdict || 'Analysis Complete'}
                  </h2>

                  <p className="verdict-deck">
                    {finalAnalysis?.executive_summary}
                  </p>

                  <div className="verdict-byline">
                    <span>By Fake News Detection AI Desk</span>
                    <span>·</span>
                    <span className="verdict-byline__confidence">
                      Confidence: {overallScore}%
                    </span>
                    <span>·</span>
                    <span>{todayStr}</span>
                  </div>
                </div>

                {/* EDITION FLAG */}
                <div className="edition-flag">
                  <span className="results-topbar__logo-dot" />
                  Special Report: Full Analysis Below
                </div>

                {/* THREE-COLUMN NEWSPAPER GRID */}
                <div className="newspaper-grid">

                  {/* ── LEFT COLUMN: ML Classifier ── */}
                  <div className="newspaper-col">
                    <div className="col-label">ML Classifier</div>

                    <div className="col-section-title">Statistical Analysis</div>
                    <div className="col-sub">
                      {mlData?.model_type || 'TF-IDF + Logistic Regression'}
                    </div>

                    {(mlData?.model_accuracy || modelInfo?.model_accuracy) && (
                      <div className="accuracy-pill">
                        ✓ {mlData?.model_accuracy || modelInfo?.model_accuracy} Accuracy
                      </div>
                    )}

                    <div className="prob-row">
                      <div className="prob-label-row">
                        <span className="fake-color">Fake Probability</span>
                        <span className="fake-color">{fakePercent}%</span>
                      </div>
                      <div className="bar-track">
                        <div className="bar-fill fake-bar" style={{ width: `${fakePercent}%` }} />
                      </div>
                    </div>

                    <div className="prob-row">
                      <div className="prob-label-row">
                        <span className="real-color">Real Probability</span>
                        <span className="real-color">{realPercent}%</span>
                      </div>
                      <div className="bar-track">
                        <div className="bar-fill real-bar" style={{ width: `${realPercent}%` }} />
                      </div>
                    </div>

                    {mlData?.top_signals && mlData.top_signals.length > 0 && (
                      <div className="signals-section">
                        <h4>Influential Word Signals</h4>
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
                      {modelInfo?.model_trained_at
                        ? `Model trained ${modelInfo.model_trained_at}`
                        : 'Model metadata unavailable'}
                    </div>
                  </div>

                  {/* Column rule */}
                  <div className="newspaper-col-rule" />

                  {/* ── CENTRE COLUMN: Main Story ── */}
                  <div className="newspaper-col">
                    <div className="col-label">In-Depth Analysis</div>

                    <div className="col-section-title">Full Investigative Report</div>

                    {finalAnalysis?.executive_summary && (
                      <div className="main-story-dropcap">
                        <p className="main-story-body">{finalAnalysis.executive_summary}</p>
                      </div>
                    )}

                    {finalAnalysis?.ml_insights && (
                      <div className="insight-pull">
                        <h4>📊 Statistical Findings</h4>
                        <p>{finalAnalysis.ml_insights}</p>
                      </div>
                    )}

                    {finalAnalysis?.news_cross_check && (
                      <div className="insight-pull">
                        <h4>🌐 News Coverage Cross-Check</h4>
                        <p>{finalAnalysis.news_cross_check}</p>
                      </div>
                    )}

                    {finalAnalysis?.red_flags && finalAnalysis.red_flags.length > 0 && (
                      <div className="red-flags-box">
                        <h4>🚩 Risk Factors / Red Flags</h4>
                        <ul>
                          {finalAnalysis.red_flags.map((flag, i) => (
                            <li key={i}>{flag}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {finalAnalysis?.recommendations && (
                      <div className="recommendation-pull">
                        <strong>💡 Recommendation: </strong>
                        {finalAnalysis.recommendations}
                      </div>
                    )}
                  </div>

                  {/* Column rule */}
                  <div className="newspaper-col-rule" />

                  {/* ── RIGHT COLUMN: Live News Sources ── */}
                  <div className="newspaper-col">
                    <div className="col-label">Live News Verification</div>

                    <div className="col-section-title">Corroborating Sources</div>
                    <div className="col-sub">NewsAPI &amp; Real-time Feeds</div>

                    <div className="articles-count-badge">
                      {newsSources.length} Source{newsSources.length !== 1 ? 's' : ''} Found
                    </div>

                    <div className="articles-scroll">
                      {newsSources.length > 0 ? (
                        newsSources.map((art, idx) => {
                          const href = safeUrl(art.url)
                          return (
                            <div key={idx} className="article-item">
                              <div className="article-header">
                                <span className="article-source">{art.source || 'News Source'}</span>
                                {art.published_at && (
                                  <span className="article-date">
                                    {new Date(art.published_at).toLocaleDateString(undefined, {
                                      month: 'short',
                                      day: 'numeric',
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
                          <p>
                            No directly matching articles were found in live news streams for this
                            exact query.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* NEWSPAPER FOOTER */}
                <div className="newspaper-footer">
                  <span>Fake News Detection · AI Fact-Check Edition</span>
                  <span>{todayStr}</span>
                  <span>Powered by Dual-Branch AI Analysis</span>
                </div>
              </div>
            )}
          </main>
        </div>
      </div>
    </>
  )
}

export default App
