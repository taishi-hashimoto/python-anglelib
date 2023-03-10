"Handles angles in various units and convensions."
import numpy as np
from typing import Union
from .wrap import wrap

class Angle:

    Degrees = "Degrees"
    Radians = "Radians"
    CCW = "CCW"
    CW = "CW"
    N = "N"
    E = "E"
    W = "W"
    S = "S"

    _zero_str2num = {
        True: {"N": 90., "E": 0., "S": -90., "W": 180.},
        False: {"N": np.pi/2., "E": 0., "S": -np.pi/2, "W": np.pi}
    }

    _direction_str2num = {"CCW": 1, "CW": -1}

    @classmethod
    def _str2val_origin(cls, zero, degrees) -> float:
        if isinstance(zero, str):
            return cls._zero_str2num[degrees][zero]
        else:
            return zero

    @classmethod
    def _str2val_direction(cls, direction) -> int:
        if isinstance(direction, str):
            return cls._direction_str2num[direction.upper()]
        else:
            assert direction in [1, -1], "direction must be 1 or -1."
            return direction

    @classmethod
    def _str2val_unit(cls, unit) -> bool:
        if isinstance(unit, str):
            unit = unit.upper()
            if unit[0].startswith(cls.Degrees[0]):  # degrees
                return True
            elif unit[0].startswith(cls.Radians[0]):  # radians
                return False
            else:
                raise ValueError(f"Unknown unit: {unit}")

    def __init__(
        self,
        value: float, *,
        zero: Union[float, str] = None,
        direction: Union[int, str] = None,
        unit: str = None
    ):
        self._scalar = np.isscalar(value)
        self._value = np.atleast_1d(value)
        if unit is not None:
            self._unit = unit
        else:
            self._unit = self.Degrees  # By default degrees
        if direction is not None:
            self._direction = direction
        else:
            self._direction = self.CCW  # By default CCW
        if zero is None:
            zero = self.E  # By default East
        self._zero = zero

    def _fix(self, value: np.ndarray, is_scalar=None):
        if is_scalar is None:
            is_scalar = self._scalar
        if is_scalar:
            return value.item()
        else:
            return value

    def to_math(self, degrees=False) -> Union[float, np.ndarray]:
        """Evaluate `zero` and `direction` so that the resultant angle follows
        the conventions in mathematics (measured CCW from East.)
        Unit is changed to radians by default.

        Parameters
        ==========
        degrees: bool
            Set `True` to convert the result into degrees.
            Default is `False`, which converts the result into radians.
        """
        x = self._fix(self._value * self.direction + self.zero)
        if self.degrees and not degrees:
            x = np.deg2rad(x)
        elif not self.degrees and degrees:
            x = np.rad2deg(x)
        return x

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self._fix(self._value)

    @property
    def degrees(self) -> bool:
        return self._str2val_unit(self._unit)

    @property
    def zero(self) -> float:
        return self._str2val_origin(self._zero, self.degrees)

    @property
    def direction(self) -> int:
        return self._str2val_direction(self._direction)

    @property
    def ccw(self) -> bool:
        return self.direction == 1

    @property
    def cw(self) -> bool:
        return not self.ccw

    @property
    def direction_str(self) -> str:
        idir = self._str2val_direction(self._direction)
        return f"{idir:+1d} ({self.CCW if self.ccw else self.CW})"

    @property
    def unit_str(self) -> str:
        return "°" if self.degrees else ""

    def convert(
        self,
        zero: float = None,
        direction: Union[int, str] = None,
        unit: str = None,
    ) -> 'Angle':
        v = self.to_math(self.degrees)

        if unit is not None:
            degrees = self._str2val_unit(unit)
            if self.degrees and not degrees:  #  deg2rad
                v = np.deg2rad(v)
            elif not self.degrees and degrees:  # rad2deg
                v = np.rad2deg(v)
        else:
            unit = self._unit
            degrees = self._str2val_unit(unit)
        if direction is not None:
            d = self._str2val_direction(direction)
        else:
            direction = self._direction
            d = self.direction
        if zero is None:
            zero = self._zero
        o = self._str2val_origin(zero, degrees)
        return Angle(
            d * (v - o),
            zero=zero, direction=direction, unit=unit)

    def wrap(self, lo=None, hi=None) -> 'Angle':
        degrees = self.degrees
        if lo is None:
            if degrees:
                lo = -180
            else:
                lo = -np.pi
        if hi is None:
            if degrees:
                hi = 180
            else:
                hi = np.pi
        value = wrap(self._fix(self._value), lo, hi)
        return Angle(value, zero=self._zero, direction=self._direction, unit=self._unit)

    def __float__(self) -> float:
        return float(self._value.item())

    def __array__(self) -> np.ndarray:
        return self._value

    def __str__(self) -> str:
        return f"{self._fix(self._value)}{self.unit_str}[0{self.unit_str} := {self._zero}{self.unit_str if not isinstance(self._zero, str) else ''}, {self.direction_str}]"
    
    def __repr__(self) -> str:
        return self.__str__()
