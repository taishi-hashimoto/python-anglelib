"""Quasi-equispaced angular grid based on the vertives of icosahedral
subdivision (icosphere).
"""

import numpy as np
from ..vecmath import direction, rotate
from ..wrap import wrap as _wrap
from ..radial import Radial


class Icosphere:
    """Quasi-equispaced angular grid based on the vertives of icosahedral
    subdivision.
    """
    def __init__(self, n):
        """
        Compute the vertices of an icosphere.
        
        Args:
            n:
                The number of division in a side of a plane of a icosahedron.
        """
        from itertools import chain
        from math import sqrt, atan2

        r3 = sqrt(3)
        r5 = sqrt(5)
        icosahedron = [
            (1, r3, (-3-r5)/2),
            (-2, 0, (-3-r5)/2),
            (1, -r3, (-3-r5)/2),
            (-(1+r5)/2, -(1+r5)*r3/2, (1-r5)/2),
            (1+r5, 0, (1-r5)/2),
            (-(1+r5)/2, (1+r5)*r3/2, (1-r5)/2),
            ((1+r5)/2, (1+r5)*r3/2, (r5-1)/2),
            (-1-r5, 0, (r5-1)/2),
            ((1+r5)/2, -(1+r5)*r3/2, (r5-1)/2),
            (-1, -r3, (3+r5)/2),
            (2, 0, (3+r5)/2),
            (-1, r3, (3+r5)/2),
        ]
        # Rotate to align two vertices on Z axis.
        icosahedron = rotate(icosahedron, "Y", -atan2(icosahedron[10][0], icosahedron[10][2]))
        # Make sure these vertices have exact zeros in X and Y coordinates.
        icosahedron[1, (0, 1)] = 0.
        icosahedron[10, (0, 1)] = 0.
        
        triangles = [
            (0, 1, 5),
            (0, 5, 6),
            (0, 6, 4),
            (0, 4, 2),
            (0, 2, 1),
            (1, 7, 5),
            (1, 3, 7),
            (1, 2, 3),
            (2, 8, 3),
            (2, 4, 8),
            (7, 11, 5),
            (5, 11, 6),
            (6, 11, 10),
            (6, 10, 4),
            (9, 10, 11),
            (7, 9, 11),
            (10, 8, 4),
            (3, 9, 7),
            (3, 8, 9),
            (8, 10, 9),
        ]
        self._ico = icosahedron
        self._tri = triangles
        points = {}
        for triangle in triangles:
            tri, ind = self.divide_triangle(icosahedron, triangle, n)

            for i, t in zip(chain.from_iterable(ind), chain.from_iterable(tri)):
                (ia, ib, ii), (jb, jc, jj) = i
                if ii == 0: # a
                    assert jj == 0
                    ib = ia
                    jb = -1
                    jc = -1
                elif ii == n and jj == 0: # b
                    ia = ib
                    ii = 0
                    jb = -1
                    jc = -1
                elif ii == n and jj == n: # c
                    ia = jc
                    ib = jc
                    ii = 0
                    jb = -1
                    jc = -1
                    jj = 0
                elif ii == n: # on bc
                    ia = jb
                    ib = jc
                    ii = jj
                    jb = -1
                    jc = -1
                    jj = 0
                elif ii == jj: # on ac
                    ib = jc
                    jb = -1
                    jc = -1
                    jj = 0

                if jj == 0: # on ab
                    jb = -1
                    jc = -1

                if ia > ib:
                    ia, ib = ib, ia
                    ii = n - ii
                if jb > jc:
                    jb, jc = jc, jb
                    jj = n - jj
                
                if jb != -1 and ia > jb:
                    ia, ib, ii, jb, jc, jj = jb, jc, jj, ia, ib, ii

                i = (ia, ib, ii), (jb, jc, jj)
                points[i] = t
        
        points = list(points.values())
        assert len(points) == 10*n**2+2

        
        self.n = n
        self.vertices = np.array(points) / np.linalg.norm(points, axis=-1, keepdims=True)
        self.average_separation = np.deg2rad(69. / n)
    
    @staticmethod
    def from_angular_separation(separation, degrees=False):
        """
        Generate 
        separation:
            Required angular separation between a closest pair in degree.
            n is automatically determined such that '69 / n < separation'.
        """
        from itertools import count
        if not degrees:
            separation = np.rad2deg(separation)
        for n in count(1):
            if 69 / n < separation:
                break
        return Icosphere(n)

    def to_radial(self):
        ze, az = direction(self.vertices).T
        r = Radial(ze=ze, az=az, radians=True)
        return r

    def to_direction(self, *, el=True, ze=False, ccw=None, degrees=False, wrap=None, sortby=None):
        """
        Return directions of vertices.
        
        If el == True, np.c_[az, el] will be returned.
        If ze == True, np.c_[ze, az] will be returned.

        wrap:
            For example, 'wrap=dict(az=(0, 360))' will limit azimuth angle within a range [0, 360).
        sortby:
            Angle name. If given, sorted by the specified angle then followed by the angle not specified.

        """
        d = direction(self.vertices)  # ze, az, ccw, radians
        is_azel = el and not ze
        if (is_azel and (ccw is None or not ccw)) or (not is_azel and (ccw is not None and not ccw)):
            d[:, 1] = np.pi/2 - d[:, 1]
        if is_azel:
            d = np.c_[d[:, 1], np.pi/2-d[:, 0]]
        if degrees:
            d = np.rad2deg(d)

        def toi(i):
            if isinstance(i, str):
                if i == "az":
                    if is_azel:
                        i = 0
                    else:
                        i = 1
                elif i == "ze":
                    i = 0
                elif i == "el":
                    i = 1
            return i

        if wrap is not None:
            for i, (left, right) in wrap.items():
                i = toi(i)
                d[:, i] = _wrap(d[:, i], left, right)
        # Sort by az.
        if sortby is not None:
            i = toi(sortby)
            if i == 0:
                key = 0, 1
            else:
                key = 1, 0
            d = np.array(sorted(d.tolist(), key=lambda x: tuple(x[i] for i in key)))
        return d
    
    def __iter__(self):
        return iter(self.vertices)
    
    def __len__(self):
        return len(self.vertices)

    @staticmethod
    def divide_triangle(vertices, triangle, n):
        """
        Divide a triangle by n, and return all vertices and indices.
        a is the top vertex.
        b is the left bottom, c is the right bottom.
        """
        ia, ib, ic = triangle
        a, b, c = (np.array(vertices[i]) for i in triangle)

        u = (b - a) / n
        v = (c - b) / n
        indices = []
        triangles = []
        for i in range(n):
            for j in range(i+1):
                # p is on a -> b
                iabi = (ia, ib, i)
                ibcj = (ib, ic, j)
                iabi1 = (ia, ib, i+1)
                ibcj1 = (ib, ic, j+1)
                ip = (iabi, ibcj)
                p = a + i*u + j*v
                # q is on a -> b and end point.
                iq = (iabi1, ibcj)
                q = p + u
                # r is on right of q
                ir = iabi1, ibcj1
                r = q + v
                indices.append((ip, iq, ir))
                triangles.append((p, q, r))
                # s is right of p
                js = iabi, ibcj1
                s = p + v
                if j != i:
                    indices.append((ip, ir, js))
                    triangles.append((p, r, s))
        return triangles, indices
