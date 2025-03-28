"""
Follows Pytorch3D conventions

.. code-block:: python

    M = [
            [Rxx, Ryx, Rzx, 0],
            [Rxy, Ryy, Rzy, 0],
            [Rxz, Ryz, Rzz, 0],
            [Tx,  Ty,  Tz,  1],
        ]

To apply the transformation to points, which are row vectors, the latter are
converted to homogeneous (4D) coordinates and right-multiplied by the M matrix:

.. code-block:: python

    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    [transformed_points, 1] ∝ [points, 1] @ M

See https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/transform3d.py
"""
import numpy as np
import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    # gltf uses quaternions in the order [x, y, z, w], but we need to convert it to [w, x, y, z] for Pytorch3D
    quaternion = torch.Tensor(np.concatenate([quaternion[3:], quaternion[:3]]))
    rotation_matrix = np.eye(4)
    # TODO: it is strange that it needs the transpose
    rotation_matrix[:3, :3] = quaternion_to_matrix(quaternion).numpy().transpose()
    return rotation_matrix




def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).

    From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    """
    From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d
    """
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).

    From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def matrix_to_euler_angles_numpy_deg(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to Euler angles in degrees.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Euler angles in degrees as tensor of shape (..., 3).
    """
    matrix = matrix[..., :3, :3]
    return np.degrees(matrix_to_euler_angles(torch.Tensor(matrix), "XYZ").numpy())


def matrix_to_euler_angles_and_translation_numpy(matrix: np.ndarray) -> np.ndarray:
    """
    Output the Euler angles and translation from the input matrix
    """
    euler_angles = matrix_to_euler_angles_numpy_deg(matrix)
    translation = matrix[..., 3, :3]
    return np.stack([euler_angles, translation])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

        From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    From https://github.com/facebookresearch/pytorch3d/blob/fe0b1bae49e7144021a9eb63169e855f51dd4dd3/pytorch3d/transforms/rotation_conversions.py#L196
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def euler_angles_to_matrix_numpy(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    return euler_angles_to_matrix(torch.Tensor(euler_angles), convention).numpy()
