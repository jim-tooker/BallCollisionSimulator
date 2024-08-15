"""
##Parameter classes for the Ball Collision Simulator system.
"""
from typing import Tuple, List
from dataclasses import dataclass
import vpython as vp
from ballcollide.ball_sim_enums import CollisionType

__author__ = "Jim Tooker"


class PhysicsParameters:
    """
    Class to store the physical parameters of a ball.
    """
    def __init__(self, mass: float, position: Tuple[float, float], velocity: Tuple[float, float]):
        """
        Args:
            mass (float): Mass of the ball in kg.
            position (Tuple[float, float]): Initial position of the ball (x, y) in meters.
            velocity (Tuple[float, float]): Initial velocity of the ball (vx, vy) in m/s.
        """
        self.mass: float = mass
        self.position: vp.vector = vp.vector(*position, 0)
        self.velocity: vp.vector = vp.vector(*velocity, 0)


@dataclass
class BallParameters:
    """
    Data class to store all parameters of a ball, including physics and visual properties.

    Attributes:
        physics_params (PhysicsParameters): The physical parameters of the ball.
        color (vp.vector): Color of the ball.
        name (str): Name or identifier for the ball.
    """

    physics_params: PhysicsParameters
    color: vp.vector
    name: str


@dataclass
class SimParameters:
    """
    Data class to store all parameters for a Simulator instance.

    Attributes:
        ball_params (List[BallParameters]): List of parameters for each ball.
        simulation_time (float): Total time to simulate.
        collision_type (ballcollide.ball_sim_enums.CollisionType): Type of collision to simulate:
                                        (elastic, inelastic, or partially elastic).
        cor (float): Coefficient of Restitution (used for partially elastic collisions)
    """

    ball_params: List[BallParameters]
    simulation_time: float
    collision_type: CollisionType
    cor: float
