# Weighted Structural Equation Recovery via IDP Importance Weights

## The Problem

We want to recover the structural equation for a node Y in a linear SCM:

```
Y = β₁·PA₁ + β₂·PA₂ + ... + βₖ·PAₖ + ε
```

where PA₁...PAₖ are the observed parents of Y. Naive OLS on observational data is biased when latent confounders exist between parents and Y — which is exactly the setting FCI/PAGs are designed for.

## The Key Insight

IDP already solves the hard part: it gives us an identification formula for P(Y | do(X₁, X₂, ...)) that expresses the interventional distribution purely in terms of observational quantities. Our `evaluate_causal_effect` computes per-data-point importance weights w_i that reweight observational data to match this interventional distribution.

**If we run IDP with treatment = {all observed parents of Y} and it's identifiable**, then these weights transform observational data into pseudo-interventional data where:

1. The parents are effectively exogenous (intervention removes confounding)
2. The structural equation Y = Σ βₖ·PAₖ + ε holds by definition
3. The noise ε is independent of the parents under intervention

This means **weighted least squares with IDP weights is a consistent estimator of the true structural coefficients**.

## Why It Works (More Formally)

In observational data, we observe from P(Y, PA₁, ..., PAₖ). Confounders create spurious correlations between PA and ε, biasing OLS.

IDP gives us weights w_i such that for any function g:

```
E[g(Y, PA) under do(PA)] = Σᵢ wᵢ · g(Yᵢ, PAᵢ) / Σᵢ wᵢ
```

Under do(PA), the structural equation Y = Σ βₖ·PAₖ + ε holds with ε ⊥ PA. This is exactly the condition for OLS consistency. So WLS with weights wᵢ gives:

```
β̂_WLS →  true β  as n → ∞
```

The weights come from the identification formula — they're ratios of densities derived from the do-calculus rules that IDP applies. We already compute them in `evaluate.py`'s `resolve()` function.

## The Algorithm

1. From the PAG, identify the **definite parents** of Y (nodes X with X → Y, i.e., tail at X, arrow at Y)
2. Run IDP with `treatment = [all definite parents]`, `outcome = [Y]`
3. If identifiable: extract the per-data-point importance weights (before binning)
4. Run weighted OLS: `Y ~ PA₁ + PA₂ + ... + PAₖ` using those weights
5. The coefficients are the recovered structural equation

## When It Fails

- **Joint effect not identifiable:** IDP may identify P(Y|do(X₁)) but not P(Y|do(X₁, X₂)). In that case we can't recover the full equation — we can only report the marginal effect of individual parents.
- **Non-linear SCM:** WLS gives a linear approximation. The R² tells you how good that approximation is.
- **Circle endpoints in PAG:** We may not know the true parents with certainty. Possible parents (o→) might or might not belong in the equation.
- **Latent parents:** If a true parent is unobserved, it's absorbed into ε. The equation is correct for observed variables but incomplete.

## What's Novel Here

The individual pieces exist — IDP identification, importance weighting, WLS. But combining IDP importance weights with multivariate WLS to recover structural coefficients from a PAG is, as far as we can tell, new. The standard approach in the literature is either:

- Assume no latent confounders (use naive OLS on the DAG)
- Use instrumental variables (requires finding valid instruments)
- Use do-calculus for point identification of single effects

This approach uses do-calculus (via IDP) to enable **simultaneous recovery of all structural coefficients** while properly handling latent confounding.

## Two Output Modes

1. **Full structural equation of Y:** Use all definite parents from the PAG. Requires joint identifiability of P(Y | do(PA(Y))).
2. **Marginal causal effect:** For a selected treatment X, show the single-coefficient marginal effect E[Y|do(X)] = β·X + α. This uses the existing single-treatment IDP result.

Both are shown in the UI. The full structural equation is the primary goal; the marginal effect is a fallback and provides context for the selected treatment-outcome pair.
