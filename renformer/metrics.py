"""
Evaluation metrics for REnFormer (paper Section 5.3).

All point metrics are computed on non-trivial timesteps (raw generation
> EPSILON_MW = 0.1 MW) to avoid inflating accuracy with trivially correct
nighttime zeros.
"""
import numpy as np

try:
    from scipy.special import ndtr as _normal_cdf
    from scipy.stats import norm as _normal
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

EPSILON_MW = 0.1


def _normal_pdf(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def _normal_cdf_fallback(z):
    return 0.5 * (1.0 + np.sign(z) * (1.0 - np.exp(-0.147 * z ** 2 * (1 + 0.0765 * z ** 2))))


def active_mask(y_raw: np.ndarray) -> np.ndarray:
    """Boolean mask: True where raw generation exceeds the intermittency threshold."""
    return y_raw > EPSILON_MW


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray = None,
) -> float:
    """
    Closed-form CRPS for Gaussian predictive distribution
    (Gneiting & Raftery 2007, Eq. 21):

        CRPS(N(μ,σ²), y) = σ [z(2Φ(z)−1) + 2φ(z) − 1/√π]
        where z = (y − μ) / σ
    """
    if mask is not None:
        y_true, mu, sigma = y_true[mask], mu[mask], sigma[mask]
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    if _HAS_SCIPY:
        Phi = _normal_cdf(z)
        phi = _normal.pdf(z)
    else:
        Phi = _normal_cdf_fallback(z)
        phi = _normal_pdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def evaluate(
    y_raw: np.ndarray,
    y_pred_mu: np.ndarray,
    y_pred_sigma: np.ndarray = None,
    denorm_mean: float = 0.0,
    denorm_std: float = 1.0,
) -> dict:
    """
    Compute MAE, RMSE (and optionally CRPS) on non-trivial timesteps.

    Parameters
    ----------
    y_raw       : raw MW values, shape (N, H, 1) or (N, H)
    y_pred_mu   : predicted means in *normalised* space, same shape
    y_pred_sigma: predicted std in normalised space (optional)
    denorm_mean, denorm_std : per-site stats to convert to MW before scoring

    When denorm_std == 1.0 (default), metrics are in normalised units —
    useful for comparing methods trained on the same normalisation.
    Pass site-level stats to get MW-scale metrics that match the paper.
    """
    y_raw  = y_raw.squeeze(-1) if y_raw.ndim == 3 else y_raw
    y_pred_mu = y_pred_mu.squeeze(-1) if y_pred_mu.ndim == 3 else y_pred_mu

    mask = active_mask(y_raw).ravel()
    y    = y_raw.ravel()
    mu   = (y_pred_mu * denorm_std + denorm_mean).ravel()

    results = {
        "MAE":  mae(y, mu, mask),
        "RMSE": rmse(y, mu, mask),
    }

    if y_pred_sigma is not None:
        sigma = y_pred_sigma.squeeze(-1) if y_pred_sigma.ndim == 3 else y_pred_sigma
        sigma_mw = (sigma * denorm_std).ravel()
        results["CRPS"] = crps_gaussian(y, mu, sigma_mw, mask)

    return results


def print_results_table(results: dict):
    """Pretty-print a {method_name: metrics_dict} table."""
    methods = list(results.keys())
    header = f"{'Method':<25} {'MAE':>8} {'RMSE':>8} {'CRPS':>8}"
    print(header)
    print("-" * len(header))
    for m, metrics in results.items():
        mae_v  = f"{metrics['MAE']:.4f}"  if 'MAE'  in metrics else "  ——  "
        rmse_v = f"{metrics['RMSE']:.4f}" if 'RMSE' in metrics else "  ——  "
        crps_v = f"{metrics['CRPS']:.4f}" if 'CRPS' in metrics else "  ——  "
        print(f"{m:<25} {mae_v:>8} {rmse_v:>8} {crps_v:>8}")
