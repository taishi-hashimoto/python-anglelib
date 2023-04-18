from anglelib.vecmath import radial_range, direction
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
    fig.tight_layout()
    plt.show()
