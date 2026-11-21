"""Missingness simulation (MCAR and MAR) for CalibMiss."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MARRuleConfig:
    """Configuration for MAR missingness rule."""

    target_feature: str
    depends_on: str
    rule_type: str
    params: dict[str, Any]


def inject_mcar(X: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject Missing Completely At Random (MCAR) missingness.

    Args:
        X: Complete feature matrix
        rate: Target missing rate (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column)
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between 0 and 1")

    rng = np.random.RandomState(seed)
    X_missing = X.copy()

    # Generate missing mask for each value independently
    mask = rng.random(X.shape) < rate
    X_missing = X_missing.mask(mask)

    # Compute realized missing rates per column
    realized = {}
    for col in X.columns:
        realized[col] = float(X_missing[col].isna().mean())

    return X_missing, realized


def inject_mcar_block(
    X: pd.DataFrame,
    rate: float,
    seed: int,
    max_block: int = 10,
    tol: float = 0.005,
    max_iter: int = 20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject MCAR missingness as whole consecutive-day blocks per ticker.

    Real daily data goes missing in streaks (halts, API gaps, delist days), not
    cell-by-cell. Each ticker is processed independently: a day opens a streak
    with probability `p`, then the masked gap is extended to a block whose
    length is drawn uniformly from 2..max_block (trimmed to the remaining rows).
    `p` is calibrated by scaling so the realized missing rate matches `rate`.

    Args:
        X: Complete feature matrix with a MultiIndex (ticker, date).
        rate: Target row-missing rate (0 to 1).
        seed: Random seed for reproducibility.
        max_block: Upper bound on streak length. A block of length 2 already
            destroys information MCAR at the same rate cannot.
        tol: Acceptable absolute drift of realized rate from target.
        max_iter: Cap on calibration iterations.

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column).
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between 0 and 1")
    if "ticker" not in X.index.names:
        raise ValueError("MCAR-BLOCK requires a MultiIndex with a 'ticker' level")
    if max_block < 2:
        raise ValueError("max_block must be at least 2")

    # Block start probability. A block of mean length L masks ~p*L rows, so start
    # from p = rate / mean_length, then rescale toward the realized rate.
    mean_length = (2 + max_block) / 2
    p = min(rate / mean_length, 0.5) if rate > 0 else 0.0

    def apply(p: float) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        values = X.copy()
        for _, ticker_frame in X.groupby(level="ticker", sort=False):
            rows = ticker_frame.index
            n = len(rows)
            if n == 0:
                continue
            mask = np.zeros(n, dtype=bool)
            starts = rng.random(n) < p
            length = rng.randint(2, max_block + 1, size=n)
            for pos in np.flatnonzero(starts):
                if pos >= n - 1:
                    continue  # no room for a length-2+ block; skip to avoid 1-cell streaks
                mask[pos : min(pos + int(length[pos]), n)] = True
            mask_2d = np.broadcast_to(mask[:, None], ticker_frame.shape)
            values.loc[rows, :] = np.where(mask_2d, np.nan, ticker_frame.to_numpy())
        return values

    if rate == 0:
        return X.copy(), {col: 0.0 for col in X.columns}
    for _ in range(max_iter):
        values = apply(p)
        realized = float(values.isna().mean().mean())
        if abs(realized - rate) <= tol:
            break
        p = max(min(p * rate / max(realized, 1e-9), 1.0), 0.0)
    realized_rates = {col: float(values[col].isna().mean()) for col in X.columns}
    return values, realized_rates


def inject_mar(
    X: pd.DataFrame,
    rule: MARRuleConfig,
    target_rate: float,
    seed: int,
    depends_on_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject Missing At Random (MAR) missingness based on observed variable.

    Uses quantile_threshold rule: missing probability = base_prob +
    (max_prob - base_prob) * I(depends_on > quantile_threshold).
    Calibrates base_prob/max_prob to achieve target_rate overall.

    Args:
        X: Complete feature matrix
        rule: MAR rule configuration
        target_rate: Target missing rate for target_feature (0 to 1)
        seed: Random seed for reproducibility
        depends_on_threshold: Pre-computed threshold for the depends_on column.
            Pass the training-split threshold so calibrate/test missingness does
            not depend on each split's own distribution. If None, computed from
            this frame's quantile.

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column)
    """
    if rule.rule_type != "quantile_threshold":
        raise ValueError(f"Unknown rule_type: {rule.rule_type}")
    if not 0 <= target_rate <= 1 or not np.isfinite(target_rate):
        raise ValueError("target_rate must be finite and between 0 and 1")
    if X.empty:
        raise ValueError("X must not be empty")

    if rule.target_feature not in X.columns:
        raise ValueError(f"Target feature {rule.target_feature} not in DataFrame")

    if rule.depends_on not in X.columns:
        raise ValueError(f"Depends on feature {rule.depends_on} not in DataFrame")

    rng = np.random.RandomState(seed)
    X_missing = X.copy()

    # Extract parameters
    base_prob = rule.params.get("base_prob", 0.05)
    max_prob = rule.params.get("max_prob", 0.5)
    quantile_threshold = rule.params.get("quantile_threshold", 0.7)
    if (
        not np.isfinite(base_prob)
        or not np.isfinite(max_prob)
        or not 0 <= base_prob <= max_prob <= 1
    ):
        raise ValueError("base_prob and max_prob must satisfy 0 <= base_prob <= max_prob <= 1")
    if not np.isfinite(quantile_threshold) or not 0 <= quantile_threshold <= 1:
        raise ValueError("quantile_threshold must be finite and between 0 and 1")

    observed = X[rule.depends_on].to_numpy()
    if not np.isfinite(observed).all():
        raise ValueError("depends_on values must be finite")
    threshold = (
        float(depends_on_threshold)
        if depends_on_threshold is not None
        else float(np.quantile(observed, quantile_threshold))
    )
    high_group = observed > threshold
    high_share = float(high_group.mean())
    if target_rate in (0.0, 1.0) or high_share in (0.0, 1.0):
        probs = np.full(len(X), target_rate)
    else:
        separation = min(target_rate / high_share, (1.0 - target_rate) / (1.0 - high_share))
        low_prob = target_rate - high_share * separation
        high_prob = target_rate + (1.0 - high_share) * separation
        probs = np.where(high_group, high_prob, low_prob)

    # Generate missing mask
    missing_mask = rng.random(len(X)) < probs

    # Apply missingness only to target feature
    X_missing[rule.target_feature] = X_missing[rule.target_feature].mask(missing_mask)

    # Compute realized missing rates per column
    realized = {}
    for col in X.columns:
        realized[col] = float(X_missing[col].isna().mean())

    return X_missing, realized


def _streak_mask(
    n: int, p: float, rng: np.random.RandomState, max_block: int
) -> np.ndarray:
    """Boolean mask of consecutive-day streaks (length drawn 2..max_block).

    A block start opens at each position with probability `p`; the block is
    extended to a random length. Positions at the end with no room for a
    length-2+ block are skipped to avoid 1-cell streaks.
    """
    mask = np.zeros(n, dtype=bool)
    starts = rng.random(n) < p
    length = rng.randint(2, max_block + 1, size=n)
    for pos in np.flatnonzero(starts):
        if pos >= n - 1:
            continue
        mask[pos : min(pos + int(length[pos]), n)] = True
    return mask


def inject_mnar(
    X: pd.DataFrame,
    rate: float,
    seed: int,
    target_feature: str,
    quantile_threshold: float = 0.7,
    base_prob: float = 0.05,
    max_prob: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject Not-Missing-At-Random missingness.

    Missingness depends on the value that is itself missing: the probability
    that `target_feature` is missing increases with that feature's own value
    (e.g. high-volatility days have their volatility missing more often). Uses
    a quantile_threshold step, mirroring MAR but with depends_on == target.
    Only `target_feature` is masked; the base/max probabilities are calibrated
    so the realized rate matches `rate`.

    Args:
        X: Complete feature matrix.
        rate: Target missing rate for target_feature (0 to 1).
        seed: Random seed for reproducibility.
        target_feature: Column to mask.
        quantile_threshold: Quantile above which the higher missing probability
            applies.
        base_prob: Missing probability for values below the threshold.
        max_prob: Missing probability for values above the threshold.

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column).
    """
    if not 0 <= rate <= 1 or not np.isfinite(rate):
        raise ValueError("rate must be finite and between 0 and 1")
    if target_feature not in X.columns:
        raise ValueError(f"target_feature {target_feature} not in DataFrame")
    if (
        not np.isfinite(base_prob)
        or not np.isfinite(max_prob)
        or not 0 <= base_prob <= max_prob <= 1
    ):
        raise ValueError("base_prob and max_prob must satisfy 0 <= base_prob <= max_prob <= 1")
    if not np.isfinite(quantile_threshold) or not 0 <= quantile_threshold <= 1:
        raise ValueError("quantile_threshold must be finite and between 0 and 1")

    values = X[target_feature].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("target_feature values must be finite")
    rng = np.random.RandomState(seed)
    X_missing = X.copy()

    threshold = float(np.quantile(values, quantile_threshold))
    high_group = values > threshold
    high_share = float(high_group.mean())
    if rate in (0.0, 1.0) or high_share in (0.0, 1.0):
        probs = np.full(len(X), rate)
    else:
        separation = min(rate / high_share, (1.0 - rate) / (1.0 - high_share))
        low_prob = rate - high_share * separation
        high_prob = rate + (1.0 - high_share) * separation
        probs = np.where(high_group, high_prob, low_prob)

    missing_mask = rng.random(len(X)) < probs
    X_missing[target_feature] = X_missing[target_feature].mask(missing_mask)

    realized = {col: float(X_missing[col].isna().mean()) for col in X.columns}
    return X_missing, realized


def inject_mcar_block_market(
    X: pd.DataFrame,
    rate: float,
    seed: int,
    max_block: int = 10,
    tol: float = 0.005,
    max_iter: int = 20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """MCAR-BLOCK with market-wide correlated outages.

    A vendor outage or exchange halt makes whole days missing for every ticker
    at once, which per-ticker-independent streak blocks cannot represent. This
    models the correlated component alone: on an outage day, every column of
    every ticker is missing. Independent per-ticker streaks remain covered by
    `inject_mcar_block`.

    Args:
        X: Complete feature matrix with a MultiIndex (ticker, date).
        rate: Target missing rate (0 to 1).
        seed: Random seed for reproducibility.
        max_block: Upper bound on streak length.
        tol: Acceptable absolute drift of realized rate from target.
        max_iter: Cap on calibration iterations.

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column).
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between 0 and 1")
    if "ticker" not in X.index.names:
        raise ValueError("MCAR-BLOCK requires a MultiIndex with a 'ticker' level")
    if max_block < 2:
        raise ValueError("max_block must be at least 2")
    if rate == 0:
        return X.copy(), {col: 0.0 for col in X.columns}

    date_codes, uniques = pd.factorize(pd.Index(X.index.get_level_values("date")), sort=True)
    n_dates = int(date_codes.max()) + 1
    date_codes = date_codes.astype(int)
    assert len(uniques) == n_dates

    mean_length = (2 + max_block) / 2
    p = min(rate / mean_length, 0.5)

    def apply(p: float) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        # Market-wide outage days: every ticker fully missing. The streak mask
        # over dates is monotone in p (fixed seed), so realized rate is too.
        date_mask = _streak_mask(n_dates, p, rng, max_block)
        market_rows = date_mask[date_codes]
        return pd.DataFrame(
            np.where(
                np.broadcast_to(market_rows[:, None], X.shape),
                np.nan,
                X.to_numpy(),
            ),
            index=X.index,
            columns=X.columns,
        )

    # Best-so-far tracking: the block structure makes achievable rates discrete,
    # so the multiplicative update can oscillate around the target. Return the
    # p whose realized rate is closest to `rate` rather than the last iterate.
    best_p, best_err, best_values = p, float("inf"), None
    for _ in range(max_iter):
        values = apply(p)
        realized = float(values.isna().mean().mean())
        err = abs(realized - rate)
        if err < best_err:
            best_err, best_p, best_values = err, p, values
        if err <= tol:
            break
        p = max(min(p * rate / max(realized, 1e-9), 1.0), 0.0)
    values = best_values if best_values is not None else apply(best_p)
    realized_rates = {col: float(values[col].isna().mean()) for col in X.columns}
    return values, realized_rates


def simulate_missingness_grid(
    X: pd.DataFrame,
    mechanisms: list[str],
    rates: list[float],
    seeds: list[int],
    mar_rules: dict[str, MARRuleConfig],
) -> dict[str, dict[float, dict[int, tuple[pd.DataFrame, dict[str, float]]]]]:
    """Simulate missingness grid across all combinations.

    Args:
        X: Complete feature matrix
        mechanisms: List of mechanisms ("MCAR", "MAR")
        rates: List of missing rates
        seeds: List of random seeds
        mar_rules: Dict mapping mechanism to MAR rule config

    Returns:
        Nested dict: results[mechanism][rate][seed] = (X_missing, realized)
    """
    results: dict[str, dict[float, dict[int, tuple[pd.DataFrame, dict[str, float]]]]] = {}

    for mechanism in mechanisms:
        results[mechanism] = {}
        for rate in rates:
            results[mechanism][rate] = {}
            for seed in seeds:
                if mechanism == "MCAR":
                    X_miss, realized = inject_mcar(X, rate, seed)
                elif mechanism == "MCAR-BLOCK":
                    X_miss, realized = inject_mcar_block(X, rate, seed)
                elif mechanism == "MCAR-BLOCK-MARKET":
                    X_miss, realized = inject_mcar_block_market(X, rate, seed)
                elif mechanism == "MNAR":
                    if "MNAR" not in mar_rules:
                        raise ValueError("MNAR rule config required for MNAR mechanism")
                    rule = mar_rules["MNAR"]
                    X_miss, realized = inject_mnar(
                        X,
                        rate,
                        seed,
                        target_feature=rule.target_feature,
                        quantile_threshold=rule.params.get("quantile_threshold", 0.7),
                        base_prob=rule.params.get("base_prob", 0.05),
                        max_prob=rule.params.get("max_prob", 0.5),
                    )
                elif mechanism == "MAR":
                    if "MAR" not in mar_rules:
                        raise ValueError("MAR rule config required for MAR mechanism")
                    rule = mar_rules["MAR"]
                    X_miss, realized = inject_mar(X, rule, rate, seed)
                else:
                    raise ValueError(f"Unknown mechanism: {mechanism}")

                results[mechanism][rate][seed] = (X_miss, realized)

    return results
