import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Use the new combined endpoint that mirrors the CLI flow
      const res = await fetch('/api/run_prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, page_size: 5 })
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`Server error: ${res.status} ${txt}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-root">
      <header>
        <h1>Fake News predictor</h1>
      </header>

      <main>
        <form onSubmit={handleSubmit}>
          <label htmlFor="text">Paste a claim or article text to verify against NewsAPI:</label>
          <textarea id="text" value={text} onChange={(e) => setText(e.target.value)} rows={8} />
          <div className="actions">
            <button type="submit" disabled={loading || !text.trim()}>
              {loading ? 'Checking...' : 'Check article'}
            </button>
          </div>
        </form>

        {error && <div className="error">Error: {error}</div>}

        {result && (
          <section className="result">
            <div className={`verdict verdict-${result.verdict || 'unknown'}`}>
              <span className="verdict-label">{result.verdict || 'unknown'}</span>
              <p>{result.message || 'No result available.'}</p>
            </div>

            {result.article ? (
              <article className="article-card">
                <h2>{result.article.title || 'Matched article'}</h2>
                <p className="meta">
                  <span>{result.article.source || 'Unknown source'}</span>
                  <span>Confidence: {(Number(result.article.score) * 100).toFixed(0)}%</span>
                </p>
                <p className="summary">{result.article.paragraph || 'No article summary available.'}</p>
                {result.article.url && (
                  <a className="article-link" href={result.article.url} target="_blank" rel="noreferrer">
                    Open article
                  </a>
                )}
              </article>
            ) : (
              <p className="no-match">No closely matching article was found for this input.</p>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
