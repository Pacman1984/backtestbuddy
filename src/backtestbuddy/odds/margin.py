"""Remove the bookmaker margin from decimal odds.

Implied probabilities ``1 / odds`` sum to more than 1; the excess is the
overround. Each method maps one market (a row of odds) to fair
probabilities that sum to 1:

- ``multiplicative``: divide implied probabilities by their sum. Spreads
  the margin in proportion to each outcome's probability.
- ``additive``: subtract an equal share of the overround from every
  outcome. Long shots with a large margin can go negative.
- ``power``: raise implied probabilities to the exponent ``k`` that makes
  them sum to 1. Takes more margin from long shots.
- ``shin``: Shin (1993) model with a share ``z`` of insider money. Also
  takes more margin from long shots; common for tennis and racing.

Power and Shin fit the favourite-longshot bias better than
multiplicative: favourites get a higher fair probability, long shots a
lower one.
"""

from typing import Callable, Dict, Literal, Sequence, Union

import numpy as np

MarginMethod = Literal["multiplicative", "additive", "power", "shin"]
MARGIN_METHODS = ("multiplicative", "additive", "power", "shin")
_BISECTION_STEPS = 100


def _multiplicative(implied: np.ndarray) -> np.ndarray:
    return implied / implied.sum(axis=1, keepdims=True)


def _additive(implied: np.ndarray) -> np.ndarray:
    n_outcomes = implied.shape[1]
    overround = implied.sum(axis=1, keepdims=True) - 1.0
    return implied - overround / n_outcomes


def _power(implied: np.ndarray) -> np.ndarray:
    """Solve ``sum q_i ** k = 1`` for ``k`` per row by bisection.

    ``sum q_i ** k`` falls in ``k`` because every ``q_i < 1``. It is
    ``n > 1`` as ``k -> 0``, so the root is bracketed once ``hi`` is
    doubled until the sum drops below 1.
    """
    lo = np.full(len(implied), 1e-9)
    hi = np.ones(len(implied))
    while True:
        too_low = (implied ** hi[:, None]).sum(axis=1) > 1.0
        if not too_low.any():
            break
        hi[too_low] *= 2.0
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        above = (implied ** mid[:, None]).sum(axis=1) > 1.0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    k = (lo + hi) / 2.0
    return implied ** k[:, None]


def _shin_probabilities(implied: np.ndarray, z: np.ndarray) -> np.ndarray:
    total = implied.sum(axis=1, keepdims=True)
    z = z[:, None]
    root = np.sqrt(z**2 + 4.0 * (1.0 - z) * implied**2 / total)
    return (root - z) / (2.0 * (1.0 - z))


def _shin(implied: np.ndarray) -> np.ndarray:
    """Solve Shin's insider share ``z`` per row by bisection.

    The probability sum falls in ``z``; it is ``sqrt(S) > 1`` at ``z = 0``
    when the book has a margin. Books without a margin (``S <= 1``) have
    no insider share and fall back to multiplicative.
    """
    has_margin = implied.sum(axis=1) > 1.0
    result = _multiplicative(implied)
    if not has_margin.any():
        return result
    book = implied[has_margin]
    lo = np.zeros(len(book))
    hi = np.full(len(book), 0.99)
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        above = _shin_probabilities(book, mid).sum(axis=1) > 1.0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    result[has_margin] = _shin_probabilities(book, (lo + hi) / 2.0)
    return result


_METHODS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}


def remove_margin(
    odds: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
    method: MarginMethod = "multiplicative",
) -> np.ndarray:
    """Convert decimal odds to fair probabilities without the margin.

    Args:
        odds: Decimal odds of one market, shape ``(n_outcomes,)``, or of
            many markets, shape ``(n_markets, n_outcomes)``.
        method: ``multiplicative``, ``additive``, ``power``, or ``shin``.

    Returns:
        Fair probabilities with the same shape as ``odds``; each row sums
        to 1. Rows with any odds ``<= 1`` or not finite are NaN.

    Raises:
        ValueError: If ``method`` is unknown or a market has fewer than
            two outcomes.

    Example:
        >>> remove_margin([1.9, 1.9])
        array([0.5, 0.5])
        >>> remove_margin([1.5, 2.5], method="additive")
        array([0.63333333, 0.36666667])
    """
    if method not in _METHODS:
        raise ValueError(
            f"method must be one of {MARGIN_METHODS}, got {method!r}"
        )
    values = np.asarray(odds, dtype=float)
    single = values.ndim == 1
    rows = np.atleast_2d(values)
    if rows.shape[1] < 2:
        raise ValueError("A market needs at least two outcomes")

    valid = np.all(np.isfinite(rows) & (rows > 1.0), axis=1)
    fair = np.full(rows.shape, np.nan)
    if valid.any():
        fair[valid] = _METHODS[method](1.0 / rows[valid])
    return fair[0] if single else fair
