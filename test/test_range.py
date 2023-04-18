from anglelib.vecmath import radial_range, direction, radial_line_around, angle_between, radial
from anglelib import Radial, Angle
import numpy as np
import matplotlib.pyplot as plt


def test_radial_range():
    ze0, az0 = np.deg2rad([30, 45])
    ze1, az1 = np.deg2rad([60, 280])
    start = Radial(ze=ze0, az=az0, radians=True)
    stop = Radial(ze=ze1, az=az1, radians=True)
    u = Radial.range(start, stop, Angle(1, unit=Angle.Degrees), include_end=True)
    v = radial_range(ze0, az0, ze1, az1, np.deg2rad(1), include_end=True)
    assert np.allclose(u.vector, v)
    ze, az = direction(v).T
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(az0, np.rad2deg(ze0), "ks")
    ax.plot(az1, np.rad2deg(ze1), "rs")
    ax.plot(az, np.rad2deg(ze), "C0.-")
    ax.set_rlim(0, 90)
    fig.tight_layout()
    plt.show()


def test_radial_line_around():
    ze0, az0 = np.deg2rad([10, 45])
    v = radial_line_around(ze0, az0, offset=np.deg2rad(20), step=np.deg2rad(6), angle=np.deg2rad(5))
    center = Radial(ze=ze0, az=az0, radians=True)
    angle = Angle(5, unit=Angle.Degrees, direction=Angle.CW)
    print(angle.convert(direction=Angle.CCW).value)
    u = Radial.line_around(center, offset=Angle(20, unit=Angle.Degrees), step=Angle(6, unit=Angle.Degrees), angle=Angle(5, unit=Angle.Degrees))
    assert np.allclose(u.vector, v)
    a = angle_between(radial(ze0, az0), v[0])
    b = angle_between(radial(ze0, az0), v[-1])
    print(a, b)
    assert np.isclose(abs(a), abs(b))
    ze, az = direction(v).T
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(az0, np.rad2deg(ze0), "ks")
    ax.plot(az, np.rad2deg(ze), "C0.-")
    ax.plot(az[0], np.rad2deg(ze[0]), "gs")
    ax.plot(az[-1], np.rad2deg(ze[-1]), "rs")
    ax.set_rlim(0, 90)
    fig.tight_layout()
    plt.show()