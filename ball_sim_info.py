"""
##Classes to store information/state for the Ball Collision Simulator system.
"""
from typing import List, Optional
from dataclasses import dataclass
import vpython as vp
from ballcollide.ball_sim_enums import Balls, BallTrajectories
from ballcollide.ball_sim import Ball

__author__ = "Jim Tooker"


@dataclass
class CollisionInfo:
    """
    Data class to store information about a collision.

    Attributes:
        time (float): Time of collision.
        ball1 (Optional[ballcollide.ball_sim.Ball]): Ball 1 object (for elastic or partially elastic collisions)
        ball2 (Optional[ballcollide.ball_sim.Ball]): Ball 2 object (for elastic or partially elastic  collisions)
        merged_ball (Optional[ballcollide.ball_sim.Ball]): Merged ball (for inelastic collision)
    """
    time: float
    ball1: Optional[Ball]
    ball2: Optional[Ball]
    merged_ball: Optional[Ball]


@dataclass
class IntersectionInfo:
    """
    Data class to store information about an intersection of ball paths.

    Attributes:
        position (vp.vector): Position vector of intersection point.
        ball1_time (float): Time when Ball 1 crossed intersection point.
        ball2_time (float): Time when Ball 2 crossed intersection point.
    """
    position: vp.vector
    ball1_time: float
    ball2_time: float


@dataclass
class SimulatorInfo:
    """
    Data class to store information about the Simulator at a point in time.

    Attributes:
        balls (List[ballcollide.ball_sim.Ball]): List of Ball objects.  
                            - Index 0 = Ball 1.  
                            - Index 1 = Ball 2.  
                            - Index 2 = Merged Ball (Optional).  
        momentum (vp.vector): Total momentum of both balls (N⋅s).
        kinetic_energy (float): Total kinetic energy of both balls (J).
        relative_speed (float): Relative speed of the two balls with respect to each other (m/s).
        distance (float): Distance of the two balls (m).
        trajectories (ballcollide.ball_sim_enums.BallTrajectories): What the balls current trajectories are:
                                         (constant, diverging, or converging).
    """
    balls: List[Ball]
    momentum: vp.vector
    kinetic_energy: float
    relative_speed: float
    distance: float
    trajectories: BallTrajectories

    @property
    def ball1(self) -> Ball:
        """
        Alias for Ball 1.

        Returns:
            ballcollide.ball_sim.Ball: An alias for Ball 1.
        """
        return self.balls[Balls.BALL1]

    @property
    def ball2(self) -> Ball:
        """
        Alias for Ball 2.

        Returns:
            ballcollide.ball_sim.Ball: An alias for Ball 2.
        """
        return self.balls[Balls.BALL2]
