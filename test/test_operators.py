from anglelib import Angle
import numpy as np

def test_operator():
    a = Angle.from_degrees(30)

    assert np.isclose(a + 0.5, np.deg2rad(30) + 0.5)

    assert np.isclose(a - 0.5, np.deg2rad(30) - 0.5)

    assert np.isclose(0.5 - a, 0.5 - np.deg2rad(30))

