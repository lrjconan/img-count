import cslab_environ

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow as tf


def random_flip_left_right(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(array_ops.pack([1.0, 1.0, uniform_random, 1.0]), 0.5)
    return array_ops.reverse(image, mirror)

def random_flip_up_down(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(array_ops.pack([1.0, uniform_random, 1.0, 1.0]), 0.5)
    return array_ops.reverse(image, mirror)

def random_hue(image, max_delta, seed=None):
    """Adjust the hue of an RGB image by a random factor.
    Equivalent to `adjust_hue()` but uses a `delta` randomly
    picked in the interval `[-max_delta, max_delta]`.
    `max_delta` must be in the interval `[0, 0.5]`.
    Args:
    image: RGB image or images. Size of the last dimension must be 3.
    max_delta: float.  Maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction
      with the graph-level seed to determine the real seeds that will be
      used in this operation. Please see the documentation of
      set_random_seed for its interaction with the graph-level random seed.
    Returns:
    3-D float tensor of shape `[height, width, channels]`.
    Raises:
    ValueError: if `max_delta` is invalid.
    """
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')

    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')

    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_hue(image, delta)


def adjust_hue(image, delta, name=None):
    """Adjust hue of an RGB image.
    This is a convenience method that converts an RGB image to float
    representation, converts it to HSV, add an offset to the hue channel, converts
    back to RGB and then back to the original data type. If several adjustments
    are chained it is advisable to minimize the number of redundant conversions.
    `image` is an RGB image.  The image hue is adjusted by converting the
    image to HSV and rotating the hue channel (H) by
    `delta`.  The image is then converted back to RGB.
    `delta` must be in the interval `[-1, 1]`.
    Args:
    image: RGB image or images. Size of the last dimension must be 3.
    delta: float.  How much to add to the hue channel.
    name: A name for this operation (optional).
    Returns:
    Adjusted image(s), same shape and DType as `image`.
    """
    with ops.op_scope([image], name, 'adjust_hue') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, dtypes.float32)

        hsv = gen_image_ops.rgb_to_hsv(flt_image)

        hue = array_ops.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
        saturation = array_ops.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
        value = array_ops.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Note that we add 2*pi to guarantee that the resulting hue is a positive
        # floating point number since delta is [-0.5, 0.5].
        hue = math_ops.mod(hue + (delta + 1.), 1.)

        hsv_altered = array_ops.concat(3, [hue, saturation, value])
        rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    return tf.image.convert_image_dtype(rgb_altered, orig_dtype)

def random_saturation(image, lower, upper, seed=None):
    """Adjust the saturation of an RGB image by a random factor.
    Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
    picked in the interval `[lower, upper]`.
    Args:
    image: RGB image or images. Size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: An operation-specific seed. It will be used in conjunction
      with the graph-level seed to determine the real seeds that will be
      used in this operation. Please see the documentation of
      set_random_seed for its interaction with the graph-level random seed.
    Returns:
    Adjusted image(s), same shape and DType as `image`.
    Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')

    # Pick a float in [lower, upper]
    saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
    return adjust_saturation(image, saturation_factor)


def adjust_saturation(image, saturation_factor, name=None):
    """Adjust saturation of an RGB image.
    This is a convenience method that converts an RGB image to float
    representation, converts it to HSV, add an offset to the saturation channel,
    converts back to RGB and then back to the original data type. If several
    adjustments are chained it is advisable to minimize the number of redundant
    conversions.
    `image` is an RGB image.  The image saturation is adjusted by converting the
    image to HSV and multiplying the saturation (S) channel by
    `saturation_factor` and clipping. The image is then converted back to RGB.
    Args:
    image: RGB image or images. Size of the last dimension must be 3.
    saturation_factor: float. Factor to multiply the saturation by.
    name: A name for this operation (optional).
    Returns:
    Adjusted image(s), same shape and DType as `image`.
    """
    with ops.op_scope([image], name, 'adjust_saturation') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, dtypes.float32)

        hsv = gen_image_ops.rgb_to_hsv(flt_image)

        hue = array_ops.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
        saturation = array_ops.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
        value = array_ops.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

        saturation *= saturation_factor
        saturation = clip_ops.clip_by_value(saturation, 0.0, 1.0)

        hsv_altered = array_ops.concat(3, [hue, saturation, value])
        rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

        return tf.image.convert_image_dtype(rgb_altered, orig_dtype)

def random_brightness(image, max_delta, seed=None):
    """Adjust the brightness of images by a random factor.
    Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
    interval `[-max_delta, max_delta)`.
    Args:
    image: An image.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    Returns:
    The brightness-adjusted image.
    Raises:
    ValueError: if `max_delta` is negative.
    """
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')

    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_brightness(image, delta)

def random_contrast(image, lower, upper, seed=None):
    """Adjust the contrast of an image by a random factor.
    Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
    picked in the interval `[lower, upper]`.
    Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    Returns:
    The contrast-adjusted tensor.
    Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
    """
    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')

    # Generate an a float in [lower, upper]
    contrast_factor = random_ops.random_uniform([], lower, upper, seed=seed)
    return adjust_contrast(image, contrast_factor)

def adjust_brightness(image, delta):
    """Adjust the brightness of RGB or Grayscale images.
    This is a convenience method that converts an RGB image to float
    representation, adjusts its brightness, and then converts it back to the
    original data type. If several adjustments are chained it is advisable to
    minimize the number of redundant conversions.
    The value `delta` is added to all components of the tensor `image`. Both
    `image` and `delta` are converted to `float` before adding (and `image` is
    scaled appropriately if it is in fixed-point representation). For regular
    images, `delta` should be in the range `[0,1)`, as it is added to the image in
    floating point representation, where pixel values are in the `[0,1)` range.
    Args:
    image: A tensor.
    delta: A scalar. Amount to add to the pixel values.
    Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.
    """
    with ops.op_scope([image, delta], None, 'adjust_brightness') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = convert_image_dtype(image, dtypes.float32)

        adjusted = math_ops.add(flt_image,
                                math_ops.cast(delta, dtypes.float32),
                                name=name)

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)


def adjust_contrast(images, contrast_factor):
    """Adjust contrast of RGB or grayscale images.
    This is a convenience method that converts an RGB image to float
    representation, adjusts its contrast, and then converts it back to the
    original data type. If several adjustments are chained it is advisable to
    minimize the number of redundant conversions.
    `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
    interpreted as `[height, width, channels]`.  The other dimensions only
    represent a collection of images, such as `[batch, height, width, channels].`
    Contrast is adjusted independently for each channel of each image.
    For each channel, this Op computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Args:
    images: Images to adjust.  At least 3-D.
    contrast_factor: A float multiplier for adjusting contrast.
    Returns:
    The contrast-adjusted image or images.
    """
    with ops.op_scope([images, contrast_factor], None, 'adjust_contrast') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = images.dtype
        flt_images = convert_image_dtype(images, dtypes.float32)

        # pylint: disable=protected-access
        adjusted = gen_image_ops._adjust_contrastv2(flt_images,
                                                    contrast_factor=contrast_factor,
                                                    name=name)
        # pylint: enable=protected-access

    return convert_image_dtype(adjusted, orig_dtype, saturate=True)
