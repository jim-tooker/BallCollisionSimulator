"""
##Enums for the Ball Collision Simulator system.
"""
from enum import Enum, IntEnum, auto

__author__ = "Jim Tooker"


class CollisionType(Enum):
    """
    Enum to indicate the type of Collision: elastic, inelastic, or partially elastic.
    """
    ELASTIC = auto()
    INELASTIC = auto()
    PARTIAL = auto()


class Balls(IntEnum):
    """
    Enum to indicate which ball is being referenced.
    """
    BALL1 = 0
    BALL2 = 1
    MERGED = 2


class BallTrajectories(Enum):
    """
    Enum to indicate the if the balls are converging, diverging, constant distance, or merged.
    """
    CONSTANT = auto()
    CONVERGING = auto()
    DIVERGING = auto()
    MERGED = auto()
