"Radial angle <-> unit vector conversion."
import numpy as np
from .angle import Angle


class Radial:
    "Radial angle <-> unit vector conversion."

    def __init__(
        self,
        *args,
        ze: float = None,
        az: float = None,
        el: float = None,
        cw: bool = None,
        ccw: bool = None,
        degrees: bool = None,
        radians: bool = None,
    ):
        if args:
            self._horz, self._vert = args
            return

        if radians is not None and degrees is None:
            degrees = not radians
        elif degrees is not None:
            pass
        else:
            raise ValueError("Unit must be specified (degrees/radians).")
        unit = Angle.Degrees if degrees else Angle.Radians
        if cw is not None:
            assert ccw is None or ccw is not cw, "cw and ccw cannot be set at the same time."
            ccw = not cw
        elif ccw is not None:
            assert cw is None or cw is not ccw, "cw and ccw cannot be set at the same time."
        else:
            ccw = None  # By default, direction of azimuth depends on which of ze/az is set.
        if ze is None and el is not None:
            self._vert = Angle(el, zero=90 if degrees else np.pi/2, direction=-1, unit=unit)
            if ccw is None:
                ccw = False
            zero = "N"
        elif ze is not None and el is None:
            self._vert = Angle(ze, unit=unit)
            if ccw is None:
                ccw = True
                zero = "E"
        else:
            self._vert = None
        if az is not None:
            self._horz = Angle(az, zero=zero, direction="CCW" if ccw else "CW", unit=unit)
        else:
            self._horz = None

    @property
    def _ze(self) -> Angle:
        if self.has_ze:
            return self._vert

    @property
    def _el(self) -> Angle:
        if self.has_el:
            return self._vert

    @property
    def _az(self) -> Angle:
        return self._horz

    @property
    def ccw(self) -> bool:
        return self._horz.ccw

    @property
    def cw(self) -> bool:
        return not self.ccw

    @property
    def degrees(self) -> bool:
        return self._horz.degrees

    @property
    def radians(self) -> bool:
        return not self.degrees

    def to_degrees(self) -> 'Radial':
        return Radial(
            self._horz.convert(unit=Angle.Degrees),
            self._vert.convert(unit=Angle.Degrees))

    def to_radians(self) -> 'Radial':
        return Radial(
            self._horz.convert(unit=Angle.Radians),
            self._vert.convert(unit=Angle.Radians))

    def to_ccw(self) -> 'Radial':
        return Radial(
            self._horz.convert(zero=Angle.E, direction=Angle.CCW),
            self._vert)

    def to_cw(self) -> 'Radial':
        return Radial(
            self._horz.convert(zero=Angle.N, direction=Angle.CW),
            self._vert)

    @property
    def vector(self):
        """Radial unit vector (= direction cosines).
        """
        ze = self.ze_rad
        az = self.az.convert(zero=Angle.E, direction=Angle.CCW, unit=Angle.Radians).eval()
        ze = np.reshape(ze, (-1, 1))
        az = np.reshape(az, (-1, 1))
        return np.column_stack((
            np.sin(ze) * np.cos(az),
            np.sin(ze) * np.sin(az),
            np.cos(ze)))

    @property
    def direction_cosines(self):
        return self.vector

    @property
    def ze(self) -> Angle:
        return self._vert.convert(zero=0, direction=1)

    @property
    def ze_rad(self) -> float:
        return self.ze.convert(unit=Angle.Radians).eval()

    @property
    def ze_deg(self) -> float:
        return self.ze.convert(unit=Angle.Degrees).eval()

    @property
    def el(self) -> Angle:
        return self._vert.convert(zero="N", direction=-1)

    @property
    def el_rad(self) -> float:
        return self.el.convert(unit=Angle.Radians).eval()

    @property
    def el_deg(self) -> float:
        return self.el.convert(unit=Angle.Degrees).eval()

    @property
    def az(self) -> Angle:
        return self._az

    @property
    def az_rad(self) -> float:
        return self.az.convert(unit=Angle.Radians).eval()

    @property
    def az_deg(self) -> float:
        return self.az.convert(unit=Angle.Degrees).eval()

    def __iter__(self):
        if self.has_ze:
            yield self._ze
        yield self._az
        if self.has_el:
            yield self._el

    @property
    def has_ze(self) -> bool:
        return self._vert.ccw

    @property
    def has_el(self) -> bool:
        return not self.has_ze

    def __str__(self) -> str:
        a = []
        if self.has_ze:
            a.append(f"ze={self._vert.value}{self._vert.unit_str}")
        a.append(f"az={self._horz.value}{self._horz.unit_str}")
        if self.has_el:
            a.append(f"el={self._vert.value}{self._vert.unit_str}")
        s = "({})".format(", ".join(a))
        if self.ccw:
            s += f" [0{self._horz.unit_str} := E, CCW]"
        else:
            s += f" [0{self._horz.unit_str} := N, CW]"
        return s

    def __repr__(self) -> str:
        return self.__str__()
