import numpy as np    # pip install numpy
import moderngl       # pip install moderngl


def autoexposure(texture: moderngl.Texture, whitelevel: float, imgw: int, imgh: int) -> float:
    """
    Calculate an autoexposure gain factor for the given texture.

    The gain is determined by analyzing a downscaled version of the texture
    (a mipmap level) to speed up the calculation. The core logic is in the
    `percentile_ae` function, which computes a gain that aims to balance
    the overall brightness of the image while preventing excessive clipping
    of highlights.

    If the texture has black borders (zero-valued rows and/or columns at the
    edges), they are ignored in the gain calculation.

    :param texture: image for which to calculate the exposure gain
    :param whitelevel: nominal (global) white level of the image
    :param imgw: width of the image within the viewport, in pixels
    :param imgh: height of the image within the viewport, in pixels
    :returns: the estimated autoexposure gain, or None if the image in the
              viewport is too small for analysis
    """
    if imgw * imgh >= 64:
        texw, texh = texture.size
        stats_level = np.log2(max(texw, texh) / 128)
        stats_level = int(max(stats_level, 0))
        texture.build_mipmaps(max_level=stats_level)
        statsw = texw // 2 ** stats_level
        statsh = texh // 2 ** stats_level
        stats = texture.read(stats_level)
        stats = np.frombuffer(stats, dtype="f4")
        stats = stats.reshape(statsh, statsw, 3)[::-1]
        stats_vp = crop_borders(stats)
        ae_gain = percentile_ae(stats_vp, whitelevel)
        return ae_gain


def percentile_ae(img: np.ndarray, whitelevel: float) -> float:
    """
    Estimate exposure gain using a center-weighted, percentile-based method.

    The gain is determined by finding the pixel value at a specific high percentile
    (currently, 99.5th) of per-pixel maximums. That is, 0.5% of pixels are allowed
    to have one or more clipped color components.

    The image is assumed to be in a nominal range of [0, whitelevel]. Values outside
    of this range are allowed, but the gain factor is defined relative to the nominal
    whitelevel. For example, if the nominal whitelevel is 1.5 and the top-0.5% value
    is 3.0, the gain will be 2.0.

    To avoid ever reducing the brightness of the image, while also not boosting noise
    in dark regions too much, the final gain is clamped to a range of [1, 32].

    :param img: input image, or part of an image; shape = (H, W, C)
    :param whitelevel: nominal (global) white level of the image
    :returns: estimated autoexposure gain; range = [1, 32]
    """
    def weighted_percentile(data, weights, percentile):
        """
        Calculate a weighted percentile. `np.percentile` does not support weights,
        so we sort the data and the corresponding weights, then find the data value
        at which the cumulative sum of weights reaches the target percentile, using
        linear interpolation.
        """
        data = data.ravel()
        weights = weights.ravel()
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        target_weight = percentile / 100.0 * total_weight
        return np.interp(target_weight, cum_weights, sorted_data)

    # Create a 2D Gaussian weight mask for center-weighting

    sigma = 24
    h, w = img.shape[:2]
    y, x = np.mgrid[-h/2:h/2, -w/2:w/2]
    weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    weights /= np.sum(weights)

    # Define a target whitelevel such that 0.5% of pixels would be clipped;
    # the target is not always reached, as the gain is restricted to [1, 32]

    clip_percentile = 0.5
    pixel_max = np.max(img, axis=2)
    target_whitelevel = weighted_percentile(pixel_max, weights, 100 - clip_percentile)
    gain = whitelevel / target_whitelevel
    gain = np.clip(gain, 1.0, 32.0)
    return gain


def crop_borders(img):
    nonzero = np.any(img != 0.0, axis=2)
    rowmask = np.any(nonzero, axis=1)
    img = img[rowmask, :]
    nonzero = np.any(img != 0.0, axis=2)
    colmask = np.any(nonzero, axis=0)
    img = img[:, colmask]
    return img
