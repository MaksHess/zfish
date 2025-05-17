import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rot(v1, v2):
    if np.allclose(v1, v2):
        return R.identity()
    axes = unit_vector(np.cross(v1, v2))
    x, y, z = axes
    phi = angle_between(v1, v2)
    return R.from_quat(
        [x * np.sin(phi / 2), y * np.sin(phi / 2), z * np.sin(phi / 2), np.cos(phi / 2)]
    )


def rot_safe(v1, v2, epsilon=1e-5):
    phi = angle_between(v1, v2)

    if phi < epsilon:
        return R.identity()
    if np.abs(phi - np.pi) < epsilon:
        return R.from_rotvec(np.array([np.pi, 0, 0]))
    axes = unit_vector(np.cross(v1, v2))
    x, y, z = axes
    return R.from_quat(
        [x * np.sin(phi / 2), y * np.sin(phi / 2), z * np.sin(phi / 2), np.cos(phi / 2)]
    )


def random_vector(dims: int = 3, seed=None):
    rng = np.random.default_rng(seed=seed)
    x = rng.standard_normal(dims)
    return x / np.linalg.norm(x)


def rotate_grid(xx, yy, zz, rotation):
    xx_rot_flat, yy_rot_flat, zz_rot_flat = rotation.apply(
        np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    ).T
    return (
        xx_rot_flat.reshape(xx.shape),
        yy_rot_flat.reshape(yy.shape),
        zz_rot_flat.reshape(zz.shape),
    )


def cart2sph(x, y, z):
    xy_squared = x**2 + y**2
    r = np.sqrt(xy_squared + z**2)  # radial
    theta = np.arctan2(x, y) + np.pi  # inclination!!
    phi = np.arctan2(np.sqrt(xy_squared), z)  # azimuth!!
    return r, theta, phi


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def draw_ellipsoid(side_length=32, a=1.0, b=1.0, c=1.0, a_direction=(1.0, 0.0, 0.0)):
    rotation = rot((1.0, 0.0, 0.0), a_direction)
    xx, yy, zz = np.meshgrid(*[np.linspace(-1, 1, side_length) for _ in range(3)])
    xx_rot, yy_rot, zz_rot = rotate_grid(xx, yy, zz, rotation)
    img = xx_rot**2 / a**2 + yy_rot**2 / b**2 + zz_rot**2 / c**2
    return img < 1


def draw_random_ellipsoid(side_length=32, axis_range=(0.6, 1.0), seed=None):
    a_direction = random_vector(seed=seed)
    rng = np.random.default_rng(seed=seed)
    axis_lengths = np.sort(rng.uniform(*axis_range, size=3))
    return draw_ellipsoid(
        side_length=side_length,
        a=axis_lengths[0],
        b=axis_lengths[1],
        c=axis_lengths[2],
        a_direction=a_direction,
    )


def field_of_ellipsoids(
    side_length_single=32, n=5, axis_range=(0.2, 1.0), direction=(1.0, 0.0, 0.0)
):
    space = np.linspace(axis_range[0], axis_range[1], n)

    canvas = np.zeros((n * side_length_single,) * 3, dtype=int)

    i = 1
    for ix, a in enumerate(space):
        for iy, b in enumerate(space):
            for iz, c in enumerate(space):
                ellipsoid = draw_ellipsoid(
                    side_length=side_length_single, a=a, b=b, c=c, a_direction=direction
                )
                x = ix * side_length_single
                y = iy * side_length_single
                z = iz * side_length_single
                canvas[
                    x : x + side_length_single,
                    y : y + side_length_single,
                    z : z + side_length_single,
                ] = ellipsoid * i
                i += 1
    return canvas
