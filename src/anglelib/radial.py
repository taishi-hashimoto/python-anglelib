"Radial angle <-> unit vector conversion."
import numpy as np
from typing import Tuple
from .angle import Angle
from .vecmath import (
    radial as _radial,
    radial_range as _range,
    direction as _direction,
    radial_line_around as _line_around
)


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
            assert isinstance(self._horz, Angle)
            assert isinstance(self._vert, Angle)
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
        return _radial(self.ze.to_math(), self.az.to_math())

    @property
    def direction_cosines(self):
        return self.vector

    @property
    def ze(self) -> Angle:
        return self._vert.convert(zero=0, direction=1)

    @property
    def ze_rad(self) -> float:
        return self.ze.convert(unit=Angle.Radians).value

    @property
    def ze_deg(self) -> float:
        return self.ze.convert(unit=Angle.Degrees).value

    @property
    def el(self) -> Angle:
        return self._vert.convert(zero="N", direction=-1)

    @property
    def el_rad(self) -> float:
        return self.el.convert(unit=Angle.Radians).value

    @property
    def el_deg(self) -> float:
        return self.el.convert(unit=Angle.Degrees).value

    @property
    def az(self) -> Angle:
        return self._az

    @property
    def az_rad(self) -> float:
        return self.az.convert(unit=Angle.Radians).value

    @property
    def az_deg(self) -> float:
        return self.az.convert(unit=Angle.Degrees).value

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

    @staticmethod
    def range(
        start: 'Radial',
        stop: 'Radial',
        step: Angle,
        axis: Tuple[float, float, float] = None,
        include_end: bool = False
    ) -> 'Radial':
        ze0 = start.ze_rad
        az0 = start.az_rad
        ze1 = stop.ze_rad
        az1 = stop.az_rad
        v = _range(
            ze0, az0,
            ze1, az1,
            step=step.to_math(),
            axis=axis,
            include_end=include_end
        )
        ze, az = _direction(v).T
        return Radial(ze=ze, az=az, radians=True)

    @staticmethod
    def line_around(
        center: 'Radial',
        offset: Angle,
        step: Angle,
        angle: Angle,
    ) -> 'Radial':
        ze0 = center.ze_rad
        az0 = center.az_rad
        v = _line_around(
            ze0, az0, offset.to_math(), step.to_math(), angle.to_math())
        ze, az = _direction(v).T
        return Radial(ze=ze, az=az, radians=True)
