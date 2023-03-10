"""Basic tools for vector mathematics.

Angles are in radian measured counterclockwise from East.
Vectors are rowwise.
"""
import math
import numpy as np
from collections.abc import Iterable
from .vecmath import normalized


def length(v, axis=-1, keepdims=False):
    return np.linalg.norm(v, axis=axis, keepdims=keepdims)


def distance(a, b, combination=False):
    """
    Calculates the distance of two vector pair a and b.
    ans(i, j) = the distance between a(i) and b(j).
    """
    a, b = np.atleast_2d(a, b)
    la = np.shape(a)[0]
    lb = np.shape(b)[0]
    if combination:
        return np.reshape(np.linalg.norm(
            np.kron(a, np.ones((lb, 1))) -
            np.kron(np.ones((la, 1)), b), axis=-1), (la, lb)).squeeze()
    else:
        return np.linalg.norm(a - b, axis=-1).squeeze()


def direction(v):
    """Directions of N vectors in radian.

    v:
        N by 2 or N by 3 array.
        Vectors in two or three dimensional space.

    Returns:
        N by 2 arrays, solid angles (zenith, azimuth).
        If v is two dimensional, zenith angles are always pi/2.
    """
    v = np.array(v)
    sz = v.shape
    if len(sz) == 1:
        v.shape = (1,) + sz
        sz = v.shape
    if sz[-1] == 2:
        return np.column_stack((
            0.5 * np.pi * np.ones(sz[0]),
            np.arctan2(v[:, 1], v[:, 0])
        ))
    elif sz[-1] == 3:
        return np.column_stack((
            np.arccos(v[:, 2] / np.linalg.norm(v, axis=-1)),
            np.arctan2(v[:, 1], v[:, 0])
        ))
    else:
        raise ValueError("Vector must have 2 (x,y) or 3 (x,y,z) elements.")


def normalized(v, axis=-1, zero="warn"):
    "Return normalized vector of v."
    lv = length(v, axis=axis, keepdims=True)
    c = lv != 0
    if zero == "warn":
        if not np.all(c):
            from warnings import warn
            warn("Normalizing zero-length vector: {}".format(v))
    elif zero == "ignore":
        pass
    else:
        raise ZeroDivisionError
    return v / np.where(c, lv, 1)


def angle_between(a, b, axis=-1, ref=None):
    """Returns the angle between two vectors.

    Args:
        a: [m, n] matrix. n vectors of size m.
        b: same as a.
        ref: A vector referred to as rotation axis.
            This is not actually used, but returned angle will be determined
            along the direction of right-screw based on this bector.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    la = length(a, axis=axis)
    lb = length(b, axis=axis)
    # Compute cosine.
    cos = np.sum(a * b, axis=axis) / (la * lb)
    # Fix domain.
    cos[cos > 1] = 1.
    cos[cos < -1] = -1.
    # Convert it to angle.
    ang = np.arccos(cos)
    # If ref is given, measure angle along it.
    if ref is not None:
        ref = np.atleast_2d(ref)
        v = np.cross(a, b, axis=axis)
        s = np.sum(v * ref, axis=axis) < 0
        ang[s] = 2*np.pi - ang[s]
    # Convert angle to scalar if both input were a single vector.
    if len(a) == 1 and len(b) == 1:
        ang = ang.item()
    return ang


def rotate(v, a1, a2, offset=None):
    """
    Rotates the vector 'v' according to a1 and a2.
    See Quaternion.get_rotator().

    :param v: Input vector.
    :param a1: Rotation axis or initial vector.
    :param a2: Rotation angle or destination vector.
    :param offset: A vector. If given, v is assumed to have an offset of this.
    :return: Rotated vector.
    """
    v = np.array(v)
    sz = list(np.shape(v))
    n = sz[-1]
    v = v.copy()
    if n < 3:
        sz[-1] += 1
        v_ = np.zeros(sz, dtype=v.dtype)
        v_[..., :2] = v
        v = v_
    q = Quaternion.get_rotator(a1, a2)
    if offset is not None:
        offset = np.array(offset)
        if len(offset) == 2:
            offset.resize(3)
        v -= offset
    v = q.rotate(v)
    if offset is not None:
        v += offset
    if n == 2:
        v = v[..., 0:n]
    return v


def orthogonal(v):
    x2 = v[:, 0]**2
    y2 = v[:, 1]**2
    z2 = v[:, 2]**2
    a = np.zeros_like(v)
    x_is_0 = np.logical_and(x2 <= y2, x2 <= z2)
    y_is_0 = np.logical_and(y2 <= x2, y2 <= z2)
    z_is_0 = ~np.logical_or(x_is_0, y_is_0)
    # X == 0
    l1 = np.sqrt(z2 + y2)
    a[x_is_0, 1] = v[:, 2] / l1
    a[x_is_0, 2] = -v[:, 1] / l1
    # Y == 0
    l2 = np.sqrt(x2 + z2)
    a[y_is_0, 0] = -v[:, 2] / l2
    a[y_is_0, 2] = v[:, 0] / l2
    # Z == 0
    l3 = np.sqrt(x2 + y2)
    a[z_is_0, 0] = v[:, 1] / l3
    a[z_is_0, 1] = -v[:, 0] / l3
    return a


def pca_normal(xyz):
    """
    Returns the normal vector from a set of 3-d points using the
    principal component analysis (PCA).

    :param xyz: N by 3 matrix.
                3-d points which are assumed to consist of a plane.
    :return: The normal vector of the surface.
    """
    assert np.ndim(xyz) == 2
    n, dim = np.shape(xyz)
    assert dim == 3
    assert n > dim
    # Subtract the average of the data.
    data = xyz - np.mean(xyz, axis=0)
    # Calculate the covariance matrix.
    cov = data.T.dot(data) / n
    # Get an eigenvector corresponding to the minimum eigenvalue.
    e, v = np.linalg.eig(cov)
    return v[:, np.argmin(e)]


class Quaternion:
    """A quaternion class."""

    def __init__(self, *args):
        """Creates a quaternion.
        Quaternion([real] [, imag])   Explicit real and/or imag.
        Quaternion(axis, angle)       Rotater."""
        nargs = len(args)
        if nargs == 0:
            self.value = np.zeros((4,))
        elif nargs == 1:
            if isinstance(args[0], Iterable):
                n = np.shape(args[0])[-1]
                if n == 4:
                    # Quaternion.
                    self.value = np.asarray(args[0])
                elif n == 3:
                    # imag only.
                    self.__resize(args[0])
                    self.imag = args[0]
                else:
                    raise ValueError("Size mismatch: {}".format(
                        args[0]))
            else:
                # Real only.
                self.__resize(args[0])
                self.real = args[0]
        elif nargs == 2:
            # Real and imag.
            self.__resize(args[0])
            self.real = args[0]
            self.imag = args[1]

    def rotate(self, v):
        """
        Rotates a vector using this quaternion.

        :param v: A vector to be rotated.
        :return: Rotated vector.
        """
        return (self * v * self.inv).imag

    def __resize(self, arg0):
        sz = np.shape(arg0)
        sz = sz[0:-1] + (4,)
        self.value = np.zeros(sz)

    @property
    def real(self):
        return self.value[..., 0]

    @real.setter
    def real(self, value):
        self.value[..., 0] = np.squeeze(value)

    @property
    def imag(self):
        return self.value[..., 1:]

    @imag.setter
    def imag(self, value):
        self.value[..., 1:] = np.squeeze(value)

    @property
    def conj(self):
        return Quaternion(self.real, -self.imag)

    @property
    def norm2(self):
        return np.sum(self.value**2)

    @property
    def inv(self):
        return self.conj / self.norm2

    @staticmethod
    def get_rotator(v, a):
        """
        Returns a quaternion that rotates 3-d points according to the following
        two parameter set:
        (1) v = axis, a = angle
            Represents a rotation by the angle 'a' around the axis 'v'.
        (2) v = org, a = tgt
            Represents a rotation that overlays the original vector 'v' onto
            the target vector 'a'.

        :param v: Rotation angle, or rotation origin.
        :param a: Rotation angle, or destination vector.
        :return: A quaternion that represents the specified rotation.
        """
        if isinstance(v, str):
            # Axis name.
            v = v.lower()
            if v == "x":
                v = [1, 0, 0]
            elif v == "y":
                v = [0, 1, 0]
            elif v == "z":
                v = [0, 0, 1]
            else:
                raise ValueError("Unknown axis name: {}".format(v))
        if isinstance(a, str):
            a = a.lower()
            if a == "x":
                a = [1, 0, 0]
            elif a == "y":
                a = [0, 1, 0]
            elif a == "z":
                a = [0, 0, 1]
            else:
                raise ValueError("Unknown axis name: {}".format(a))
        if np.size(a) == 1:
            a /= 2
            real = math.cos(a)
            imag = normalized(v) * math.sin(a)
            return Quaternion(real, imag)
        else:
            org = normalized(v)
            tgt = normalized(a)
            axis = np.cross(org, tgt)
            dot = np.dot(org, tgt)
            if dot == -1.:
                # org and tgt is in opposite direction.
                raise NotImplementedError
            else:
                s = np.sqrt((1 + dot) * 2)
                return Quaternion(s / 2, axis / s)

    def __neg__(self):
        return Quaternion(-self.real, -self.imag)

    def __add__(self, other):
        return Quaternion(self.real + other.real,
                          self.imag + other.imag)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            r1, i1 = self.real, self.imag
            r2, i2 = other.real, other.imag
            if len(r1.shape) == 1:
                r1.shape += (1,)
            if len(r2.shape) == 1:
                r2.shape += (1,)
            return Quaternion(
                ((r1 * r2).T - np.dot(i1.conjugate(), i2.T)).T,
                r1 * i2 + r2 * i1 + np.cross(i1, i2))
        else:
            return self * Quaternion(other)

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inv
        else:
            return Quaternion(self.value/other)

    def __rtruediv__(self, other):
        if isinstance(other, Quaternion):
            return other.inv * self
        else:
            return Quaternion(self.value/other)

    def __rmul__(self, other):
        return Quaternion(other) * self

    def __rdiv__(self, other):
        return other * self.inv

    def __getitem__(self, *indices):
        return self.value.__getitem__(*indices)

    def __repr__(self):
        return self.value.__repr__()
