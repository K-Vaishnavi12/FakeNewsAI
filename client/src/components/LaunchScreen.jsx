import { useCallback, useRef, useState } from 'react'
import './LaunchScreen.css'

/**
 * Full-viewport cinematic intro shown before the app itself.
 *
 * The exit is animated rather than immediate: we flip into an "exiting" state,
 * let the CSS transition run, and only then tell the parent to unmount us. The
 * timer is the authority (not `transitionend`) because a reduced-motion user
 * gets a 0ms transition that may never fire an event, and a backgrounded tab
 * can drop the event entirely -- either case would strand the user on the
 * splash forever. `firedRef` keeps the handoff idempotent so a click during
 * the fade-out cannot schedule a second `onEnter`.
 *
 * @param {{ onEnter: () => void }} props
 */
export default function LaunchScreen({ onEnter }) {
  const [exiting, setExiting] = useState(false)
  const firedRef = useRef(false)

  const enter = useCallback(() => {
    if (firedRef.current) return
    firedRef.current = true
    setExiting(true)

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    window.setTimeout(onEnter, reduced ? 0 : 600)
  }, [onEnter])

  return (
    <div
      className={`launch ${exiting ? 'launch--exiting' : ''}`}
      onClick={enter}
      role="presentation"
      data-testid="launch-screen"
    >
      {/* Decorative layers: texture, dark wash, vignette. */}
      <div className="launch__bg" aria-hidden="true" />
      <div className="launch__overlay" aria-hidden="true" />
      <div className="launch__vignette" aria-hidden="true" />

      <div className="launch__content">
        <p className="launch__badge">
          <span className="launch__badge-dot" aria-hidden="true" />
          AI-Powered News Verification
        </p>

        <h1 className="launch__title">
          Fake News <span className="launch__title-em">AI</span> Detection
        </h1>

        <div className="launch__divider" aria-hidden="true">
          <span className="launch__divider-dot" />
        </div>

        <p className="launch__tagline">
          See Beyond the Headlines. Discover the Truth.
        </p>
        <p className="launch__subtagline">
          Because Every Story Deserves the Truth.
        </p>

        <button
          type="button"
          className="launch__cta"
          onClick={(e) => {
            // The wrapper already handles the click; stop it here so `enter`
            // is not reached twice through the bubble path.
            e.stopPropagation()
            enter()
          }}
        >
          Click anywhere on screen to enter
          <span className="launch__cta-arrow" aria-hidden="true">→</span>
        </button>
      </div>

      <p className="launch__disclaimer">
        FAKE NEWS AI assists verification with retrieved evidence and
        machine-learning signals. It is not an absolute truth detector.
      </p>
    </div>
  )
}
