from typing import Union, List, ClassVar, Iterable


class vector:
    def __init__(self, x: float, y: float, z: float) -> None: ...
    def __add__(self, other: 'vector') -> 'vector': ...
    def __sub__(self, other: 'vector') -> 'vector': ...
    def __mul__(self, other: Union[float, 'vector']) -> 'vector': ...
    def __rmul__(self, other: Union[float, 'vector']) -> 'vector': ...
    def __truediv__(self, other: Union[float, 'vector']) -> 'vector': ...
    def norm(self) -> 'vector': ...
    def dot(self, other: 'vector') -> float: ...

    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, value: float) -> None: ...

    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, value: float) -> None: ...

    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, value: float) -> None: ...

    @property
    def mag(self) -> float: ...


class color:
    black: ClassVar[vector]
    white: ClassVar[vector]
    red: ClassVar[vector]
    green: ClassVar[vector]
    blue: ClassVar[vector]
    yellow: ClassVar[vector]
    cyan: ClassVar[vector]
    magenta: ClassVar[vector]
    orange: ClassVar[vector]
    purple: ClassVar[vector]

    @classmethod
    def gray(cls, luminance: float) -> vector: ...

    @classmethod
    def rgb_to_hsv(cls, v: vector) -> vector: ...

    @classmethod
    def hsv_to_rgb(cls, v: vector) -> vector: ...

    @classmethod
    def rgb_to_grayscale(cls, v: vector) -> vector: ...


class canvas:
    def __init__(
        self,
        title: str = ...,
        width: int = ...,
        height: int = ...,
        background: vector = ...,
        center: vector = ...
    ) -> None: ...

    def center(self) -> vector: ...
    def select(self) -> None: ...
    def delete(self) -> None: ...


class sphere:
    pos: vector
    visible: bool
    radius: float
    make_trail: bool

    def __init__(
        self,
        pos: vector = ...,
        radius: float = ...,
        color: vector = ...,
        size: vector = ...,
        axis: vector = ...,
        opacity: float = ...,
        shininess: vector = ...,
        emissive: bool = ...,
        visible: bool = ...,
        make_trail: bool = ...,
        up: vector = ...,
    ) -> None: ...

    def clear_trail(self) -> None: ...


class label:
    pos: vector
    visible: bool

    def __init__(
        self,
        pos: vector = ...,
        text: str = ...,
        color: vector = ...,
        align: str = ...,
        xoffset: int = ...,
        yoffset: int = ...,
        font: str = ...,
        background: vector = ...,
        opacity: float = ...,
        box: bool = ...,
        border: int = ...,
        line: bool = ...,
        linecolor: vector = ...,
        linewidth: int = ...,
        space: int = ...,
        pixel_pos: bool = ...,
        height: int = ...,
        visible: bool = ...
    ) -> None: ...


class curve:
    def __init__(
        self,
        pos: List[vector] = ...,
        color: vector = ...,
        radius: float = ...,
        size: vector = ...,
        origin: vector = ...,
        axis: vector = ...,
    ) -> None: ...

    def append(self, pos: vector) -> None: ...
    def modify(self, index: int, pos: vector) -> None: ...
    def clear(self) -> None: ...


# class methods
def mag(A: vector) -> float: ...
def norm(A: vector) -> vector: ...
def dot(A: vector, B: vector) -> float: ...
def arange(A: int, B: int, step: int) -> Iterable[int]: ...
def rate(A: int) -> None: ...