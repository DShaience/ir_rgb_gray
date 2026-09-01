"""Feature extraction for IR / RGB / grayscale image classification."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import stats as scipy_stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Fixed order used both when training the stage-B model and when scoring a
# new image -- keep train.py and classify.py in sync via this constant.
FEATURE_NAMES = [
    "entropy",
    "skewness",
    "kurtosis",
    "frac_near_black",
    "frac_near_white",
    "laplacian_var",
    "edge_density",
    "bright_area_frac",
    "bright_largest_blob_frac",
    "dark_area_frac",
    "dark_largest_blob_frac",
    "mean_intensity",
    "std_intensity",
    # GLCM texture, FFT power-spectrum shape, and MSCN (BRISQUE-style local
    # contrast normalization) statistics -- empirically raised cross-validated
    # accuracy from 89% to 95% (clean) / 90% (under quality-degradation
    # augmentation), with the biggest gain on the grayscale class.
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "lbp_entropy",
    "lbp_uniform_frac",
    "fft_high_freq_ratio",
    "fft_power_slope",
    "mscn_skew",
    "mscn_kurtosis",
    "mscn_var",
]


def load_image(path: str) -> tuple[np.ndarray, dict]:
    """Load an image as 8-bit BGR, plus metadata about how it was stored.

    Handles single-channel, BGRA, and >8-bit (e.g. radiometric TIFF) sources.
    16-bit+ data is percentile-stretched to 8-bit since we only need it for
    photometric/texture statistics, not calibrated temperature values.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not read image: {path}")

    meta = {"orig_dtype": str(raw.dtype), "orig_shape": tuple(raw.shape)}

    if raw.dtype != np.uint8:
        lo, hi = np.percentile(raw, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(raw.min()), float(raw.max()) or 1.0
        raw = np.clip((raw.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if raw.ndim == 2:
        meta["channels"] = 1
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        meta["channels"] = 4
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        meta["channels"] = 3
        bgr = raw
    else:
        raise ValueError(f"unsupported image shape {raw.shape} for {path}")

    return bgr, meta


def channel_color_stats(bgr: np.ndarray) -> dict:
    """Stats that separate real color content from color-container-but-monochrome data."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    diff_rb = np.abs(r - b)
    mean_channel_diff = float(np.mean([diff_rg.mean(), diff_gb.mean(), diff_rb.mean()]))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation_mean = float(hsv[:, :, 1].mean()) / 255.0

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return {
        "mean_channel_diff": mean_channel_diff,
        "saturation_mean": saturation_mean,
        "colorfulness": colorfulness,
    }


def monochrome_features(gray: np.ndarray) -> dict:
    """Histogram/texture/blob features used to separate grayscale vs IR whitehot/blackhot."""
    gray_f = gray.astype(np.float32)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-8)
    nz = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = cv2.bitwise_not(bright_mask)
    bright_area_frac, bright_largest_blob_frac = _blob_stats(bright_mask)
    dark_area_frac, dark_largest_blob_frac = _blob_stats(dark_mask)

    feats = {
        "entropy": entropy,
        "skewness": float(scipy_stats.skew(gray_f.ravel())),
        "kurtosis": float(scipy_stats.kurtosis(gray_f.ravel())),
        "frac_near_black": float(np.mean(gray < 10)),
        "frac_near_white": float(np.mean(gray > 245)),
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "bright_area_frac": bright_area_frac,
        "bright_largest_blob_frac": bright_largest_blob_frac,
        "dark_area_frac": dark_area_frac,
        "dark_largest_blob_frac": dark_largest_blob_frac,
        "mean_intensity": float(gray_f.mean()) / 255.0,
        "std_intensity": float(gray_f.std()) / 255.0,
    }
    feats.update(_glcm_features(gray))
    feats.update(_lbp_features(gray))
    feats.update(_fft_features(gray))
    feats.update(_mscn_features(gray))
    return feats


def _glcm_features(gray: np.ndarray) -> dict:
    """Gray-level co-occurrence texture -- captures pixel-pair relationships
    that plain edge/blob stats miss (e.g. how correlated neighboring
    intensities are), which turned out to be the single strongest feature
    for telling real grayscale photos apart from thermal imagery.
    """
    q = (gray.astype(np.float32) / 256 * 32).astype(np.uint8)  # 32 levels keeps this cheap
    glcm = graycomatrix(q, distances=[3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def _lbp_features(gray: np.ndarray) -> dict:
    """Local Binary Pattern micro-texture: real photos tend to have richer,
    less uniform micro-texture (skin, fabric, foliage) than thermal imagery.
    """
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    nz = hist[hist > 0]
    return {
        "lbp_entropy": float(-np.sum(nz * np.log2(nz))),
        "lbp_uniform_frac": float(hist[:8].sum()),
    }


def _fft_features(gray: np.ndarray) -> dict:
    """Radial power spectrum shape. Natural photos follow a roughly 1/f^2
    power-law; blurrier/lower-resolution thermal captures fall off faster,
    so both the slope and the high-vs-low frequency energy ratio carry signal.
    """
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    max_r = min(cy, cx)

    radial = np.bincount(r.ravel(), weights=power.ravel())[:max_r]
    radial = radial / (radial.sum() + 1e-8)

    low = radial[: max_r // 4].sum()
    high = radial[3 * max_r // 4 :].sum()

    freqs = np.arange(1, max_r)
    log_p = np.log(radial[1:] + 1e-12)
    log_f = np.log(freqs)
    slope = float(np.polyfit(log_f, log_p, 1)[0]) if max_r > 1 else 0.0

    return {
        "fft_high_freq_ratio": float(high / (low + 1e-8)),
        "fft_power_slope": slope,
    }


def _mscn_features(gray: np.ndarray) -> dict:
    """BRISQUE-style mean-subtracted-contrast-normalized coefficients: local
    mean/contrast normalization isolates texture regularity independent of
    overall brightness, which raw-pixel skew/kurtosis alone can't separate.
    """
    g = gray.astype(np.float32)
    mu = cv2.GaussianBlur(g, (7, 7), 7 / 6)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(g * g, (7, 7), 7 / 6) - mu * mu))
    mscn = (g - mu) / (sigma + 1)
    return {
        "mscn_skew": float(scipy_stats.skew(mscn.ravel())),
        "mscn_kurtosis": float(scipy_stats.kurtosis(mscn.ravel())),
        "mscn_var": float(mscn.var()),
    }


def _blob_stats(mask: np.ndarray) -> tuple[float, float]:
    """Area fraction covered by `mask`, and the fraction taken by its single largest blob."""
    area_frac = float(np.mean(mask > 0))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return area_frac, 0.0
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    return area_frac, float(largest / mask.size)
