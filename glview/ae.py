import numpy as np    # pip install numpy
import moderngl       # pip install moderngl


def autoexposure(texture: moderngl.Texture, whitelevel: float, clip_pct: float) -> float:
    """
    Calculate an autoexposure gain factor for the given texture. The texture
    is intended to represent a cropped region of a full-size image, with the
    calculated gain intended to yield an ideal exposure for that region. The
    gain is defined relative to the given global whitelevel.

    The gain is determined by analyzing a downscaled version of the texture
    (a mipmap level) to speed up the calculation. If the texture has black
    borders (zero-valued rows and/or columns at the edges), they are ignored.

    :param texture: texture for which to calculate the exposure gain
    :param whitelevel: global nominal white level of the full-size image
    :returns: the estimated autoexposure gain, or None if the texture is too
              small for analysis
    """
    texw, texh = texture.size
    if texw * texh >= 64:
        stats_level = np.log2(max(texw, texh) / 128)
        stats_level = int(max(stats_level, 0))
        texture.build_mipmaps(max_level=stats_level)
        statsw = texw // 2 ** stats_level
        statsh = texh // 2 ** stats_level
        stats = texture.read(stats_level)
        stats = np.frombuffer(stats, dtype="f4")
        stats = stats.reshape(statsh, statsw, 3)[::-1]
        stats = crop_borders(stats)
        if stats.size > 16:
            ae_gain = percentile_ae(stats, whitelevel, clip_pct)
            return ae_gain


def percentile_ae(img: np.ndarray, whitelevel: float, clip_pct: float) -> float:
    """
    Estimate exposure gain using a center-weighted, percentile-based method.

    The gain is determined by finding the pixel value at a specific percentile
    of per-pixel maximums. For example, if the given clip_pct is 95, 5% of pixels
    are allowed to have one or more clipped color components.

    The image is assumed to be in a nominal range of [0, whitelevel]. Values
    outside of this range are allowed, but the gain factor is defined relative
    to the nominal white level. For example, if the nominal white level is 1.5,
    and the top-5% value representing a target white level ends up at 3.0, the
    gain will be 2.0.

    :param img: input image, or part of an image; shape = (H, W, C)
    :param whitelevel: global nominal white level of the image
    :returns: estimated gain factor to yield the target clip percentage
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

    # Define a target whitelevel such that the given percentage of pixels
    # will be clipped

    pixel_max = np.max(img, axis=2)
    target_white = weighted_percentile(pixel_max, weights, 100 - clip_pct)
    target_gain = whitelevel / target_white
    return target_gain


def crop_borders(img):
    span = lambda a: slice(a.argmax(), a.size - a[::-1].argmax())
    nonzero = np.any(img != 0.0, axis=2)
    rowmask = np.any(nonzero, axis=1)
    colmask = np.any(nonzero, axis=0)
    img = img[span(rowmask), :]
    img = img[:, span(colmask)]
    return img
