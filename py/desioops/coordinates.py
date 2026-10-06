import numpy as np

# Dictionary of common fisheye projections.
# We don't a priori assume any projection for this camera, but instead fit it from
# some found "on camera" positions of objects and their "true" positions
# from ephemerids.
# Projections take in the zenith angle (90 - altitude) and convert it
# to the linear radius on the image plane. Reverse projection
# does the opposite.
# TODO make these actual functions and not ananoymous? For docstring purposes?
projections = {"equidistant": lambda x: x, "orthographic": lambda x: np.sin(x),
               "stereographic": lambda x: 2 * np.tan(x / 2),
               "equisolid": lambda x: 2 * np.sin(x / 2)}
reverse_projections = {"equidistant": lambda x: x, "orthographic": lambda x: np.arcsin(x),
                       "stereographic": lambda x: 2 * np.arctan(x / 2),
                       "equisolid": lambda x: 2 * np.arcsin(x / 2) }

def xy_to_altaz(x, y, proj, f, dtheta):
    # TODO: docstring, x, y are expected to already be relative to image center
    # f is the "focal length" (in pixels) found by the fit.
    # dtheta is the rotation o fthe camera, also found by the fit.
    r = np.hypot(x, y)
    alt = 90 - np.rad2deg(reverse_projections[proj](r / f))

    # On the image, positive y goes down and positive x goes to the right (West,
    # but azimuth is defined relative to east which is left on the image.
    # So, atan2 gives us the normal axis (positive y -> up, positive x -> right)
    # And we rotate it by 180 degrees to get the inverted image axis,
    # which also ensures all values are positive.
    az = np.atan2(x, y)
    az = np.rad2deg(az + np.pi)

    az -= dtheta # Rotation correction
    return alt, az

def altaz_to_xy(alt, az, proj, f, dtheta):
    # TODO: docstring
    r = f * projections[proj](np.deg2rad(90 - alt))
    az_corrected = np.array(az, copy=True) + dtheta

    x = -np.sin(np.deg2rad(az_corrected)) * r
    y = -np.cos(np.deg2rad(az_corrected)) * r

    return x, y


def get_com_patch(im, x_guess, y_guess, width, thresh=False):
    patch = im[(y_guess - width // 2):(y_guess + width // 2), (x_guess - width // 2):(x_guess + width // 2)]

    if thresh:
        patch = patch > 215

    # Nothing in this patch was bright enough
    # so our guess is probably pretty bad.
    if not np.any(patch):
        return None, None

    # xx, yy are vector positions from (0,0), which is the top left corner
    xx, yy = np.meshgrid(np.arange(width), np.arange(width))

    x_com = np.sum(xx * patch) / np.sum(patch)
    y_com = np.sum(yy * patch) / np.sum(patch)
    return x_com, y_com

def find_object_com(im, x_guess, y_guess, width_1=80, width_2=60):
    x_thresh, y_thresh = get_com_patch(im, x_guess, y_guess, width_1, True)

    # Our guess was bad.
    if (x_thresh is None) and (y_thresh is None):
        return None, None

    # These need to be integers again, for slicing.
    x_guess = x_guess - (width_1 // 2) + int(x_thresh)
    y_guess = y_guess - (width_1 // 2) + int(y_thresh)

    x_com, y_com = get_com_patch(im, x_guess, y_guess, width_2, False)

    # COM is in coordinates relative to the second window size,.
    # This converts it back to relative to the whole image size.
    x_final = x_guess - (width_2 / 2) + x_com
    y_final = y_guess - (width_2 / 2) + y_com

    return x_final, y_final