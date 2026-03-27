import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


INCLUDE_ZEROS_IN_FIT = False

# ============================================================
# Candidate decay models q(p)
# ============================================================


def exp_decay(p, A, b):
    """
    Exponential decay model.

    Parameters
    ----------
    p : array-like
        QAOA depth values.
    A : float
        Overall amplitude or prefactor.
    b : float
        Exponential decay rate.

    Returns
    -------
    numpy.ndarray
        Model prediction `q(p) = A * exp(-b p)`.
    """
    return A * np.exp(-b * p)



def exp_quadratic(p, A, b):
    """
    Quadratic-in-depth exponential decay model.

    Parameters
    ----------
    p : array-like
        QAOA depth values.
    A : float
        Overall amplitude or prefactor.
    b : float
        Decay coefficient multiplying `p^2`.

    Returns
    -------
    numpy.ndarray
        Model prediction `q(p) = A * exp(-b p^2)`.
    """
    return A * np.exp(-b * (p ** 2))



def stretched_exp(p, A, b, k):
    """
    Stretched exponential decay model.

    Parameters
    ----------
    p : array-like
        QAOA depth values.
    A : float
        Overall amplitude or prefactor.
    b : float
        Scale in the exponential.
    k : float
        Stretch exponent. Values `k < 1` correspond to slower-than-exponential
        decay, while `k > 1` correspond to faster-than-exponential decay.

    Returns
    -------
    numpy.ndarray
        Model prediction `q(p) = A * exp(-b p^k)`.
    """
    return A * np.exp(-b * (p ** k))



def power_law(p, A, alpha):
    """
    Power-law decay model.

    Parameters
    ----------
    p : array-like
        QAOA depth values.
    A : float
        Overall amplitude or prefactor.
    alpha : float
        Power-law exponent.

    Returns
    -------
    numpy.ndarray
        Model prediction `q(p) = A * p^(-alpha)`.
    """
    return A * (p ** (-alpha))



def exp_over_power(p, A, b, alpha):
    """
    Mixed exponential-over-power decay model.

    Parameters
    ----------
    p : array-like
        QAOA depth values.
    A : float
        Overall amplitude or prefactor.
    b : float
        Exponential decay rate.
    alpha : float
        Power-law exponent.

    Returns
    -------
    numpy.ndarray
        Model prediction `q(p) = A * exp(-b p) / p^alpha`.
    """
    return A * np.exp(-b * p) * (p ** (-alpha))


MODEL_SPECS = [
    # name, function, initial guess builder, bounds
    (
        "exp_decay",
        exp_decay,
        lambda y0: (min(1.0, max(float(y0), 1e-12)), 0.3),
        ([0.0, 0.0], [1.0, np.inf]),
    ),
    (
        "exp_quadratic",
        exp_quadratic,
        lambda y0: (min(1.0, max(float(y0), 1e-12)), 0.05),
        ([0.0, 0.0], [1.0, np.inf]),
    ),
    (
        "stretched_exp",
        stretched_exp,
        lambda y0: (min(1.0, max(float(y0), 1e-12)), 0.05, 1.5),
        ([0.0, 0.0, 0.1], [1.0, np.inf, 5.0]),
    ),
    (
        "power_law",
        power_law,
        lambda y0: (min(1.0, max(float(y0), 1e-12)), 2.0),
        ([0.0, 0.0], [1.0, np.inf]),
    ),
    (
        "exp_over_power",
        exp_over_power,
        lambda y0: (min(1.0, max(float(y0), 1e-12)), 0.1, 1.0),
        ([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]),
    ),
]


# ============================================================
# Fit utilities
# ============================================================


@dataclass
class FitResult:
    """
    Container for one fitted decay model.

    Attributes
    ----------
    model : str
        Name of the fitted model.
    params : numpy.ndarray
        Best-fit parameter values returned by `curve_fit`.
    perr : numpy.ndarray
        One-sigma parameter uncertainties from the fit covariance matrix.
    weighted_sse : float
        Weighted sum of squared residuals, used here as a model-ranking score.
    """

    model: str
    params: np.ndarray
    perr: np.ndarray
    weighted_sse: float



def weighted_sse(y, yhat, sigma):
    """
    Compute the weighted sum of squared residuals.

    Parameters
    ----------
    y : array-like
        Observed data values.
    yhat : array-like
        Model predictions at the same `p` values.
    sigma : array-like
        Standard errors used as weights.

    Returns
    -------
    float
        `sum(((y - yhat) / sigma)^2)`. Smaller values indicate a better fit
        under the supplied weighting.
    """
    resid = (y - yhat) / sigma
    return float(np.sum(resid ** 2))



def safe_curve_fit(func, p, y, sigma, p0, bounds, maxfev=200000):
    """
    Perform a weighted nonlinear least-squares fit with SciPy.

    Parameters
    ----------
    func : callable
        Model function.
    p : array-like
        Independent variable values.
    y : array-like
        Observed response values.
    sigma : array-like
        Standard errors used for weighted fitting.
    p0 : tuple
        Initial parameter guess.
    bounds : tuple
        Lower and upper parameter bounds passed to `curve_fit`.
    maxfev : int, default=200000
        Maximum number of function evaluations.

    Returns
    -------
    tuple
        `(popt, perr)` where `popt` is the best-fit parameter vector and `perr`
        is the vector of one-sigma uncertainties.
    """
    popt, pcov = curve_fit(
        func,
        p,
        y,
        sigma=sigma,
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr



def fit_all_models(p, y, sigma):
    """
    Fit every candidate decay model and rank them by weighted SSE.

    Parameters
    ----------
    p : array-like
        Depth values used in the fit.
    y : array-like
        Aggregated empirical proportions.
    sigma : array-like
        Standard errors used as weights.

    Returns
    -------
    list[FitResult]
        Successful fits sorted from best to worst by `weighted_sse`.

    Notes
    -----
    Models that fail to converge or violate the imposed bounds are skipped.
    """
    results = []
    y0 = y[0] if len(y) else 1e-12

    for name, func, p0_builder, bounds in MODEL_SPECS:
        try:
            popt, perr = safe_curve_fit(func, p, y, sigma, p0=p0_builder(y0), bounds=bounds)
            yhat = func(p, *popt)
            w = weighted_sse(y, yhat, sigma)
            results.append(FitResult(model=name, params=popt, perr=perr, weighted_sse=w))
        except Exception:
            continue

    results.sort(key=lambda r: r.weighted_sse)
    return results



def aggregate_with_errorbars(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Aggregate per-instance monotonicity proportions into mean curves and error bars.

    The Julia experiment writes one row per problem instance, mixer, depth, and
    event type. This function groups rows by `(problem, mixer, N, p)` and then
    combines two sources of uncertainty:

    1. across-instance uncertainty:
       variability of the empirical proportions across random instances;
    2. within-instance binomial uncertainty:
       Monte Carlo noise from a finite number of random schedules.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw Julia output containing at least the columns
        `{inc_prop, dec_prop, reps, instance_seed, problem, mixer, N, p}`.
    target : {"inc_prop", "dec_prop", "mono_prop"}
        Which quantity to aggregate. `"mono_prop"` means `inc_prop + dec_prop`.

    Returns
    -------
    pandas.DataFrame
        Aggregated table with columns including:
        `mean_q`, `std_q`, `n_inst`, `se_inst`, `se_binom`, and `se_total`.

    Notes
    -----
    `se_total = sqrt(se_inst^2 + se_binom^2)` is the error bar used for fitting.
    """
    if target not in ("inc_prop", "dec_prop", "mono_prop"):
        raise ValueError("target must be inc_prop, dec_prop, or mono_prop")

    df = df.copy()

    if target == "mono_prop":
        df["q"] = df["inc_prop"].astype(float) + df["dec_prop"].astype(float)
    else:
        df["q"] = df[target].astype(float)

    n = df["reps"].astype(float)

    # Binomial SE for the chosen event probability q.
    df["se_binom"] = np.sqrt(np.clip(df["q"], 0, 1) * np.clip(1 - df["q"], 0, 1) / n)

    grp_cols = ["problem", "mixer", "N", "p"]
    agg = (
        df.groupby(grp_cols)
        .agg(
            mean_q=("q", "mean"),
            std_q=("q", "std"),
            n_inst=("instance_seed", "nunique"),
            reps=("reps", "median"),
            mean_se_binom_sq=("se_binom", lambda x: float(np.mean(np.square(x)))),
        )
        .reset_index()
    )

    agg["std_q"] = agg["std_q"].fillna(0.0)
    agg["se_inst"] = agg["std_q"] / np.sqrt(np.maximum(agg["n_inst"], 1))
    agg["se_binom"] = np.sqrt(agg["mean_se_binom_sq"])
    agg["se_total"] = np.sqrt(agg["se_inst"] ** 2 + agg["se_binom"] ** 2)

    # Prevent zero-weight pathologies in weighted fitting.
    agg["se_total"] = np.maximum(agg["se_total"], 1e-16)
    return agg


# ============================================================
# Plotting helpers
# ============================================================


def model_param_names(model_name: str):
    """
    Return readable parameter names for a given model.

    Parameters
    ----------
    model_name : str
        Name of one of the models in `MODEL_SPECS`.

    Returns
    -------
    list[str]
        Parameter names in display order.
    """
    if model_name == "exp_decay":
        return ["A", "b"]
    if model_name == "exp_quadratic":
        return ["A", "b"]
    if model_name == "stretched_exp":
        return ["A", "b", "k"]
    if model_name == "power_law":
        return ["A", "alpha"]
    if model_name == "exp_over_power":
        return ["A", "b", "alpha"]
    return [f"theta_{i+1}" for i in range(10)]



def format_fit_text(fit: FitResult) -> str:
    """
    Format fitted parameters and uncertainties for in-plot annotation.

    Parameters
    ----------
    fit : FitResult
        Fitted model record.

    Returns
    -------
    str
        Multiline annotation containing the model name, parameter values,
        parameter uncertainties, and weighted SSE.
    """
    names = model_param_names(fit.model)
    lines = [f"model = {fit.model}"]
    for name, val, err in zip(names, fit.params, fit.perr):
        lines.append(f"{name} = {val:.3e} ± {err:.3e}")
    lines.append(f"weighted SSE = {fit.weighted_sse:.3g}")
    return "\n".join(lines)



def plot_with_fit(sub: pd.DataFrame, fit: FitResult, func, title: str, outpath: str, y_floor: float):
    """
    Plot an aggregated monotonicity curve together with its best-fit model.

    Parameters
    ----------
    sub : pandas.DataFrame
        Aggregated rows for one `(problem, mixer, N)` slice.
    fit : FitResult
        Best-fit model result.
    func : callable
        Best-fit model function.
    title : str
        Plot title.
    outpath : str
        Output path for the saved PNG.
    y_floor : float
        Positive floor used only for log-scale visualization and for stabilizing
        extremely small error bars.

    Returns
    -------
    None
    """
    p = sub["p"].to_numpy(dtype=float)
    y = sub["mean_q"].to_numpy(dtype=float)
    sigma = sub["se_total"].to_numpy(dtype=float)

    # For display only, clip values away from zero on the log axis.
    y_plot = np.maximum(y, y_floor)

    plt.figure(figsize=(8.5, 5.8))
    plt.errorbar(p, y_plot, yerr=sigma, fmt="o", capsize=3, label="mean ± SE")

    p_grid = np.linspace(float(np.min(p)), float(np.max(p)), 400)
    y_grid = func(p_grid, *fit.params)
    y_grid_plot = np.maximum(y_grid, y_floor)
    plt.plot(p_grid, y_grid_plot, "-", label=f"best fit: {fit.model}")

    plt.yscale("log")
    plt.xlabel("p (QAOA depth)")
    plt.ylabel("proportion")
    plt.title(title)
    plt.legend()

    fit_text = format_fit_text(fit)
    ax = plt.gca()
    ax.text(
        0.98,
        0.98,
        fit_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



def model_function_by_name(name: str):
    """
    Return the model function associated with a model name.

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    callable
        Matching model function.

    Raises
    ------
    KeyError
        If `name` is not in `MODEL_SPECS`.
    """
    for nm, func, _, _ in MODEL_SPECS:
        if nm == name:
            return func
    raise KeyError(name)


# ============================================================
# Main driver
# ============================================================


def main():
    """
    Command-line entry point.

    This routine loads one or more raw CSV files produced by the Julia
    monotonicity experiment, aggregates proportions and error bars, fits several
    candidate decay models, saves one plot per `(metric, problem, mixer, N)`,
    and writes a CSV summary of all successful fits.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, nargs="+", help="One or more raw CSV files from qaoa_monotonicity_extended.jl")
    ap.add_argument("--outdir", default="fits", help="Output directory")
    ap.add_argument(
        "--metric",
        default="all",
        choices=["inc", "dec", "mono", "both", "all"],
        help="Which metric to fit: inc, dec, mono=inc+dec, both=inc+dec separately, all=inc+dec+mono",
    )
    ap.add_argument("--min_p", type=int, default=2, help="Minimum p to include")
    ap.add_argument("--max_p", type=int, default=None, help="Maximum p to include")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and concatenate CSV inputs.
    dfs = []
    for path in args.csv:
        d = pd.read_csv(path)
        if len(d) == 0:
            print(f"WARNING: {path} has header but no data rows; skipping.")
            continue
        dfs.append(d)
    if not dfs:
        raise SystemExit("No usable input rows found across provided CSV files.")
    df = pd.concat(dfs, ignore_index=True)

    metrics = []
    if args.metric in ("inc", "both", "all"):
        metrics.append("inc_prop")
    if args.metric in ("dec", "both", "all"):
        metrics.append("dec_prop")
    if args.metric in ("mono", "all"):
        metrics.append("mono_prop")

    fit_rows = []

    for target in metrics:
        agg = aggregate_with_errorbars(df, target)

        agg = agg[agg["p"] >= args.min_p]
        if args.max_p is not None:
            agg = agg[agg["p"] <= args.max_p]

        median_reps = float(np.median(agg["reps"].to_numpy())) if len(agg) else 1.0
        y_floor = 0.5 / median_reps

        for (problem, mixer, N), sub in agg.groupby(["problem", "mixer", "N"], sort=True):
            sub = sub.sort_values("p")
            p = sub["p"].to_numpy(dtype=float)
            y = sub["mean_q"].to_numpy(dtype=float)
            sigma = sub["se_total"].to_numpy(dtype=float)

            if np.all(y <= 0):
                print(f"Skipping {problem} | {mixer} | N={N} | {target}: all zeros")
                continue

            mask = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
            if not INCLUDE_ZEROS_IN_FIT:
                mask = mask & (y > 0)

            p_fit, y_fit, s_fit = p[mask], y[mask], sigma[mask]

            # Avoid putting extreme weight on ultra-small standard errors.
            s_fit = np.maximum(s_fit, y_floor)

            if len(p_fit) < 4:
                print(f"Skipping {problem} | {mixer} | N={N} | {target}: insufficient points after masking")
                continue

            results = fit_all_models(p_fit, y_fit, s_fit)
            if not results:
                print(f"Skipping {problem} | {mixer} | N={N} | {target}: all model fits failed")
                continue

            best = results[0]
            best_func = model_function_by_name(best.model)

            title = f"{problem} | {mixer} | N={N} | {target}"
            outpath = os.path.join(args.outdir, f"fit_{target}_{problem}_{mixer}_N{int(N)}.png")
            plot_with_fit(sub, best, best_func, title, outpath, y_floor)

            for r in results:
                fit_rows.append(
                    {
                        "metric": target,
                        "problem": problem,
                        "mixer": mixer,
                        "N": int(N),
                        "model": r.model,
                        "weighted_sse": r.weighted_sse,
                        "params": ";".join([f"{x:.17g}" for x in r.params]),
                        "perr": ";".join([f"{x:.17g}" for x in r.perr]),
                        "best": r.model == best.model,
                        "n_points_fit": int(len(p_fit)),
                        "include_zeros_in_fit": bool(INCLUDE_ZEROS_IN_FIT),
                    }
                )

            print(f"{title} -> best={best.model}, weighted_sse={best.weighted_sse:.2f}")

        agg_out = os.path.join(args.outdir, f"aggregated_{target}.csv")
        agg.to_csv(agg_out, index=False)

    if fit_rows:
        fit_df = pd.DataFrame(fit_rows)
        fit_df.sort_values(
            ["metric", "problem", "mixer", "N", "best", "weighted_sse"],
            ascending=[True, True, True, True, False, True],
            inplace=True,
        )
        fit_out = os.path.join(args.outdir, "fit_summary.csv")
        fit_df.to_csv(fit_out, index=False)
        print(f"Wrote: {fit_out}")


if __name__ == "__main__":
    main()
