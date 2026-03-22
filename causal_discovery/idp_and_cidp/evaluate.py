"""V2: Regression-based causal effect estimation from IDP/CIDP formulas.

Replaces V1's frequency-table arithmetic with:
- Per-data-point log-importance-weights computed from the identification formula
- Regression-based density estimation for continuous conditionals
- Monte Carlo marginalization (average over observed values, no binning of intermediates)
- Discretization only at the end, for display: treatment bins + per-treatment outcome bins

Only two user-facing parameters: treatment_bins and outcome_bins.
No hidden intermediate resolution parameters.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from sklearn.linear_model import LinearRegression


def _interv_key(intervset: list[str]) -> str:
    """Convert an intervention set list to its Qop dict key."""
    return "obs" if len(intervset) == 0 else ",".join(intervset)


def _is_continuous(series: pd.Series, threshold: int = 20) -> bool:
    """Check if a Series likely represents continuous data."""
    return series.dtype.kind == "f" and series.nunique() > threshold


def _parse_query(query: str) -> tuple[list[str], list[str]]:
    """Parse treatment and outcome variable names from the query string.

    Examples:
        "P_{x}(y)"           -> (["x"], ["y"])
        "P_{x1,x2}(y,z)"    -> (["x1","x2"], ["y","z"])
        "P_{w,x1,x2}(y | z)" -> (["w","x1","x2"], ["y"])
    """
    m = re.match(r"P(?:_\{([^}]+)\})?\(([^|)]+)", query)
    if m:
        treatment = (
            [v.strip() for v in m.group(1).split(",")]
            if m.group(1)
            else []
        )
        outcome = [v.strip() for v in m.group(2).split(",")]
        return treatment, outcome
    return [], []


# ---------------------------------------------------------------------------
# Density estimation helpers
# ---------------------------------------------------------------------------

def _conditional_log_density(
    data: pd.DataFrame,
    target_vars: list[str],
    cond_vars: list[str],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate log P(target_vars | cond_vars) at each data point.

    Continuous targets: linear regression + Gaussian residuals.
    Discrete targets with discrete conditions: weighted frequency counting.
    Discrete targets with continuous conditions: treated as numeric for regression.
    """
    n = len(data)
    log_densities = np.zeros(n)

    for var in target_vars:
        series = data[var]
        use_regression = _is_continuous(series) or any(
            _is_continuous(data[cv]) for cv in cond_vars
        )

        if use_regression:
            X = data[cond_vars].apply(pd.to_numeric, errors="coerce").values
            y = series.values.astype(float)

            model = LinearRegression()
            if weights is not None:
                safe_w = np.maximum(weights, 0.0)
                total = safe_w.sum()
                if total > 0:
                    safe_w = safe_w / total * n
                else:
                    safe_w = np.ones(n)
                model.fit(X, y, sample_weight=safe_w)
                predicted = model.predict(X)
                residuals = y - predicted
                sigma = max(
                    np.sqrt(np.average(residuals**2, weights=safe_w)), 1e-10
                )
            else:
                model.fit(X, y)
                predicted = model.predict(X)
                sigma = max(np.std(y - predicted), 1e-10)

            log_densities += norm.logpdf(y, loc=predicted, scale=sigma)
        else:
            # All-discrete: weighted frequency counting via transform
            temp = data[cond_vars + [var]].copy()
            temp["_w"] = np.maximum(weights, 0.0) if weights is not None else 1.0
            joint_sum = temp.groupby(cond_vars + [var], observed=True)[
                "_w"
            ].transform("sum")
            marginal_sum = temp.groupby(cond_vars, observed=True)[
                "_w"
            ].transform("sum")
            prob = np.where(marginal_sum > 0, joint_sum / marginal_sum, 1e-10)
            log_densities += np.log(np.maximum(prob, 1e-300))

    return log_densities


def _marginal_log_density(
    data: pd.DataFrame,
    variables: list[str],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate log P(variables) at each data point."""
    n = len(data)
    any_continuous = any(_is_continuous(data[v]) for v in variables)

    if any_continuous:
        vals = (
            data[variables].apply(pd.to_numeric, errors="coerce").values.T
        )
        try:
            if weights is not None:
                safe_w = np.maximum(weights, 0.0)
                total = safe_w.sum()
                if total > 0:
                    kde = gaussian_kde(vals, weights=safe_w / total)
                else:
                    kde = gaussian_kde(vals)
            else:
                kde = gaussian_kde(vals)
            densities = kde(vals)
            return np.log(np.maximum(densities, 1e-300))
        except (np.linalg.LinAlgError, ValueError):
            return np.full(n, np.log(1.0 / n))
    else:
        # Discrete marginal via transform
        temp = data[variables].copy()
        temp["_w"] = np.maximum(weights, 0.0) if weights is not None else 1.0
        group_sum = temp.groupby(variables, observed=True)["_w"].transform(
            "sum"
        )
        total = temp["_w"].sum()
        if total > 0:
            prob = group_sum / total
        else:
            prob = pd.Series(np.ones(n) / n)
        return np.log(np.maximum(prob.values, 1e-300))


# ---------------------------------------------------------------------------
# Qop variable introspection
# ---------------------------------------------------------------------------

def _get_child_variables(qop: dict, key: str) -> list[str]:
    """Get all variables involved in a Qop entry."""
    entry = qop[key]
    if entry == 1:
        return []
    op_type = entry["type"]
    param = entry["param"]

    if op_type == "none":
        return list(param["var"])
    elif op_type == "prop6":
        return list(param["num.var"])
    elif op_type == "prop7":
        vars_set: set[str] = set()
        for sub_key in ("num.prod1", "num.prod2", "den"):
            child_key = _interv_key(param[sub_key])
            vars_set.update(_get_child_variables(qop, child_key))
        return list(vars_set)
    elif op_type == "sumset":
        child_key = _interv_key(param["interv"])
        child_vars = _get_child_variables(qop, child_key)
        return [v for v in child_vars if v not in param["sumset"]]
    elif op_type == "frac_cond":
        return _get_child_variables(qop, param["prob"])
    return []


def _detect_conditioning_vars(
    qop: dict,
    query: str,
    treatment_vars: list[str],
    outcome_vars: list[str],
) -> list[str]:
    """Detect conditioning variables from a frac_cond operation (CIDP)."""
    entry = qop[query]
    if not isinstance(entry, dict) or entry.get("type") != "frac_cond":
        return []
    den_sumset = set(entry["param"]["den.sumset"])
    child_vars = _get_child_variables(qop, entry["param"]["prob"])
    exclude = set(treatment_vars) | set(outcome_vars) | den_sumset
    return [v for v in child_vars if v not in exclude]


# ---------------------------------------------------------------------------
# Weight extraction (for WLS structural equation recovery)
# ---------------------------------------------------------------------------

def compute_importance_weights(
    qop: dict[str, Any],
    query: str,
    data: pd.DataFrame,
) -> np.ndarray:
    """Compute per-data-point importance weights from an IDP/CIDP formula.

    These weights reweight observational data to match the interventional
    distribution specified by the identification formula.  Useful for
    weighted least squares regression to recover structural coefficients.

    Returns
    -------
    np.ndarray
        Non-negative importance weights, one per data row.
    """
    n = len(data)
    cache: dict[str, np.ndarray] = {}

    def _weights_for_density(log_w: np.ndarray) -> np.ndarray:
        centered = log_w - np.median(log_w)
        centered = np.clip(centered, -500, 500)
        return np.exp(centered)

    def resolve(key: str) -> np.ndarray:
        if key in cache:
            return cache[key]

        entry = qop[key]
        if entry == 1:
            cache[key] = np.zeros(n)
            return cache[key]

        op_type = entry["type"]
        param = entry["param"]

        if op_type == "none":
            log_weights = np.zeros(n)
        elif op_type == "prop6":
            parent_key = _interv_key(param["interv"])
            if parent_key == "obs" or parent_key not in qop:
                log_parent = np.zeros(n)
                density_weights = None
            else:
                log_parent = resolve(parent_key)
                density_weights = _weights_for_density(log_parent)

            den_vars = param["den.var"]
            den_cond = param["den.cond"]
            if den_cond:
                log_density = _conditional_log_density(
                    data, den_vars, den_cond, density_weights
                )
            else:
                log_density = _marginal_log_density(
                    data, den_vars, density_weights
                )
            log_weights = log_parent - log_density
        elif op_type == "prop7":
            lw1 = resolve(_interv_key(param["num.prod1"]))
            lw2 = resolve(_interv_key(param["num.prod2"]))
            lwd = resolve(_interv_key(param["den"]))
            log_weights = lw1 + lw2 - lwd
        elif op_type == "sumset":
            log_weights = resolve(_interv_key(param["interv"]))
        elif op_type == "frac_cond":
            log_weights = resolve(param["prob"])
        else:
            raise ValueError(f"Unknown Qop type: {op_type}")

        cache[key] = log_weights
        return log_weights

    log_weights = resolve(query)

    # Convert to non-negative weights with numerical stability
    log_weights = log_weights - np.median(log_weights)
    log_weights = np.clip(log_weights, -500, 500)
    weights = np.exp(log_weights)

    # Clip extreme weights (top 1% outliers)
    p99 = np.percentile(weights, 99)
    if p99 > 0:
        weights = np.minimum(weights, p99 * 10)

    return weights


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_causal_effect(
    qop: dict[str, Any],
    query: str,
    data: pd.DataFrame,
    treatment_vars: list[str] | None = None,
    outcome_vars: list[str] | None = None,
    treatment_bins: int = 10,
    outcome_bins: int = 10,
) -> pd.DataFrame:
    """Evaluate a Qop identification formula against observational data.

    Parameters
    ----------
    qop : dict
        The Qop dict from an IDP/CIDP result (``result['Qop']``).
    query : str
        The query key in qop (``result['query']``).
    data : pd.DataFrame
        Observational data whose columns match the variable names used in qop.
    treatment_vars : list[str] or None
        Treatment variable names.  Auto-detected from the query string if None.
    outcome_vars : list[str] or None
        Outcome variable names.  Auto-detected from the query string if None.
    treatment_bins : int
        Number of bins for continuous treatment variables.
    outcome_bins : int
        Number of bins for continuous outcome variables.

    Returns
    -------
    pd.DataFrame
        Probability table with variable columns and a ``prob`` column.
        Probabilities sum to 1 within each treatment (+ conditioning) group.
    """
    # Auto-detect treatment/outcome from query string
    if treatment_vars is None or outcome_vars is None:
        auto_t, auto_o = _parse_query(query)
        if treatment_vars is None:
            treatment_vars = auto_t
        if outcome_vars is None:
            outcome_vars = auto_o

    conditioning_vars = _detect_conditioning_vars(
        qop, query, treatment_vars, outcome_vars
    )

    weights = compute_importance_weights(qop, query, data)

    return _bin_and_aggregate(
        data,
        weights,
        treatment_vars,
        outcome_vars,
        conditioning_vars,
        treatment_bins,
        outcome_bins,
    )


def _bin_and_aggregate(
    data: pd.DataFrame,
    weights: np.ndarray,
    treatment_vars: list[str],
    outcome_vars: list[str],
    conditioning_vars: list[str],
    treatment_bins: int,
    outcome_bins: int,
) -> pd.DataFrame:
    """Bin treatment/outcome, aggregate weights, normalize per group."""
    all_vars = list(treatment_vars) + list(outcome_vars) + list(conditioning_vars)
    if not all_vars:
        return pd.DataFrame({"prob": [1.0]})

    temp = data[all_vars].copy()
    temp["_w"] = weights

    # Bin treatment variables
    for tv in treatment_vars:
        if _is_continuous(data[tv]):
            temp[tv] = pd.cut(temp[tv], bins=treatment_bins)

    # Per-group outcome binning (qcut so bins span only the relevant range)
    norm_group_cols = list(treatment_vars) + list(conditioning_vars)

    if norm_group_cols:
        has_continuous_outcome = any(
            _is_continuous(data[ov]) for ov in outcome_vars
        )
        if has_continuous_outcome:
            parts = []
            for _, grp in temp.groupby(norm_group_cols, observed=True):
                grp = grp.copy()
                for ov in outcome_vars:
                    if _is_continuous(data[ov]):
                        grp[ov] = pd.qcut(
                            grp[ov], q=outcome_bins, duplicates="drop"
                        )
                parts.append(grp)
            temp = pd.concat(parts, ignore_index=True)
    else:
        for ov in outcome_vars:
            if _is_continuous(data[ov]):
                temp[ov] = pd.qcut(
                    temp[ov], q=outcome_bins, duplicates="drop"
                )

    # Aggregate: sum weights per variable combination
    agg = temp.groupby(all_vars, observed=True)["_w"].sum().reset_index()

    # Normalize per treatment + conditioning group
    if norm_group_cols:
        group_sums = agg.groupby(norm_group_cols, observed=True)[
            "_w"
        ].transform("sum")
        agg["_w"] = np.where(group_sums > 0, agg["_w"] / group_sums, 0.0)
    else:
        total = agg["_w"].sum()
        if total > 0:
            agg["_w"] /= total

    return agg.rename(columns={"_w": "prob"})
