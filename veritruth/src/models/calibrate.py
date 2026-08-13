"""Step 4a — probability calibration and the 0-100 trust score.

Two calibration strategies are supported:

* ``temperature``  — single-parameter scaling fitted on validation logits.
  Used for the transformer (works on raw logits, no refitting of the model).
* ``sigmoid``      — Platt scaling via ``CalibratedClassifierCV`` semantics,
  implemented directly on probabilities so it works for any estimator.

The fitted parameters plus the band thresholds are persisted to
``models/threshold.json`` so inference never needs the validation data.

Run::

    python -m src.models.calibrate
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.config import (
    BAND_REAL_MIN,
    BAND_SUSPICIOUS_MIN,
    THRESHOLD_PATH,
    band_for_score,
    ensure_dirs,
    get_logger,
)

LOG = get_logger("veritruth.models.calibrate")

_EPS = 1e-6


@dataclass
class Calibration:
    """Persisted calibration parameters.

    ``method`` is one of ``temperature``, ``sigmoid`` or ``identity``.
    For ``temperature`` we store ``temperature``; for ``sigmoid`` we store the
    logistic coefficients ``a`` (slope) and ``b`` (intercept) applied to the
    logit of the raw probability.
    """

    method: str = "identity"
    temperature: float = 1.0
    a: float = 1.0
    b: float = 0.0
    band_real_min: float = BAND_REAL_MIN
    band_suspicious_min: float = BAND_SUSPICIOUS_MIN
    source_model: str = "unknown"
    val_brier_before: float = 0.0
    val_brier_after: float = 0.0

    # ---------------------------------------------------------------- apply
    def apply_proba(self, proba: np.ndarray | float) -> np.ndarray:
        """Calibrate P(REAL) values that are already probabilities."""
        p = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0 - _EPS)
        if self.method == "identity":
            return p
        logit = np.log(p / (1.0 - p))
        if self.method == "temperature":
            temp = self.temperature if self.temperature > _EPS else 1.0
            return _sigmoid(logit / temp)
        return _sigmoid(self.a * logit + self.b)

    def apply_logit(self, logit: np.ndarray | float) -> np.ndarray:
        """Calibrate a raw logit / log-odds margin (transformer path)."""
        z = np.asarray(logit, dtype=float)
        if self.method == "temperature":
            temp = self.temperature if self.temperature > _EPS else 1.0
            return _sigmoid(z / temp)
        if self.method == "sigmoid":
            return _sigmoid(self.a * z + self.b)
        return _sigmoid(z)

    def trust_score(self, proba_real: float) -> float:
        """Map calibrated P(REAL) to a 0-100 trust score."""
        return round(float(np.clip(proba_real, 0.0, 1.0)) * 100.0, 2)

    def band(self, score: float) -> str:
        if score >= self.band_real_min:
            return "Real"
        if score >= self.band_suspicious_min:
            return "Suspicious"
        return "Fake"

    # ------------------------------------------------------------ persistence
    def save(self, path: Path = THRESHOLD_PATH) -> Path:
        ensure_dirs()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        LOG.info("Saved calibration (%s) to %s", self.method, path)
        return path

    @classmethod
    def load(cls, path: Path = THRESHOLD_PATH) -> "Calibration":
        """Load calibration; falls back to an identity calibration on any error."""
        if not path.exists():
            LOG.info("No calibration at %s; using identity.", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass API
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception as exc:  # pragma: no cover - corrupt file
            LOG.warning("Bad calibration file %s (%s); using identity.", path, exc)
            return cls()


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a single temperature by golden-section search on NLL.

    Deterministic, dependency-free and cannot diverge — safer than gradient
    descent for a one-parameter problem on tiny validation sets.
    """
    z = np.asarray(logits, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    if z.size == 0 or len(set(y.tolist())) < 2:
        return 1.0

    def nll(temp: float) -> float:
        p = np.clip(_sigmoid(z / max(temp, _EPS)), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    lo, hi = 0.05, 10.0
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    for _ in range(80):
        if nll(c) < nll(d):
            hi, d = d, c
            c = hi - inv_phi * (hi - lo)
        else:
            lo, c = c, d
            d = lo + inv_phi * (hi - lo)
        if abs(hi - lo) < 1e-4:
            break
    return float(round((lo + hi) / 2.0, 6))


def fit_platt(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Platt scaling: logistic regression on the logit of the raw probability."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(proba, dtype=float).ravel(), _EPS, 1.0 - _EPS)
    y = np.asarray(y_true, dtype=int).ravel()
    if p.size < 4 or len(set(y.tolist())) < 2:
        return 1.0, 0.0
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception as exc:  # pragma: no cover
        LOG.warning("Platt scaling failed (%s); identity used.", exc)
        return 1.0, 0.0


def calibrate_baseline(save: bool = True) -> Calibration:
    """Fit Platt scaling for the persisted TF-IDF baseline on the val split."""
    from src.data.split import load_splits
    from src.models.baseline import load_baseline

    artifact = load_baseline()
    val = load_splits()["val"]
    texts = val["text"].astype(str).tolist()
    y = val["label"].to_numpy(dtype=int)

    raw = artifact.predict_proba(texts)
    a, b = fit_platt(raw, y)
    calib = Calibration(
        method="sigmoid",
        a=a,
        b=b,
        source_model=artifact.name,
        val_brier_before=brier(y, raw),
    )
    calib.val_brier_after = brier(y, calib.apply_proba(raw))
    if save:
        calib.save()
    return calib


def main() -> Calibration:
    calib = calibrate_baseline(save=True)
    print("\n--- STEP 4a VERIFICATION (calibration) ------------------------")
    print(f"Method            : {calib.method} (a={calib.a:.4f}, b={calib.b:.4f})")
    print(f"Source model      : {calib.source_model}")
    print(f"Brier before/after: {calib.val_brier_before:.4f} -> {calib.val_brier_after:.4f}")
    for demo in (0.02, 0.35, 0.55, 0.95):
        score = calib.trust_score(float(calib.apply_proba(demo)))
        print(f"  P(real)={demo:.2f} -> trust={score:6.2f} -> band={band_for_score(score)}")
    print(f"Saved to          : {THRESHOLD_PATH}")
    print("Expected          : threshold.json exists; Brier after <= before")
    print("---------------------------------------------------------------\n")
    return calib


if __name__ == "__main__":
    main()
