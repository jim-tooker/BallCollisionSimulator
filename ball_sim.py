"""
##Ball module containing the `Ball` class for the Ball Collision Simulator system.
"""
from typing import Optional
import math
import vpython as vp
from ball_sim_parameters import BallParameters

__author__ = "Jim Tooker"


class Ball:
    """
    Class representing a ball in the simulation.

    Attributes:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Position of the ball (x, y) (m).
        velocity (Tuple[float, float]): Velocity of the ball (vx, vy) (m/s).
        radius (float): Radius of the ball (m).
        name (str): Name of the Ball.
        collision_point (Optional[vp.vector]): Point on the ball where the collision occurred
        collision_point_offset (Optional[vp.vector]): Vector offset between the center of the ball
                                                      and where the collision occurred
        angle (float): Angle of ball travel (0° -> +x-axis,
                                             90° -> +y-axis,
                                             180° -> -x-axis,
                                             -90° -> -y-axis).
        speed (float): Speed of ball (m/s).
        momentum (vp.vector): Momentum of ball (mvx, mvy) (N⋅s).
        momentum_mag (float): Magnitude of the momentum (N⋅s).
        kinetic_energy (float): Kinetic energy of the ball (J).
    """

    # Flag to indicate whether the GUI should be disabled (True = no GUI)
    _no_gui = False

    def __init__(self, params: BallParameters):
        """
        Args:
            params (ballcollide.ball_sim_parameters.BallParameters): Parameters for the ball.
        """
        self.mass: float = params.physics_params.mass
        self.position: vp.vector = params.physics_params.position
        self.velocity: vp.vector = params.physics_params.velocity
        self.name: str = params.name

        self.collision_point: Optional[vp.vector] = None
        self.collision_point_offset: Optional[vp.vector] = None

        if Ball._no_gui is False:
            self._sphere: vp.sphere = vp.sphere(pos=self.position,
                                                radius=self.radius,
                                                color=params.color,
                                                make_trail=True)
            self._label: vp.label = vp.label(pos=self.position,
                                             text=params.name,
                                             height=14,
                                             color=vp.color.white,
                                             box=False,
                                             opacity=0)
            self._collision_blob: Optional[vp.ellipsoid] = None


    @property
    def radius(self) -> float:
        """
        Calculate and return the radius of the ball.
        In this sim, radius is proportional to the mass (0.5m for 1 kg)
        """
        return float(0.5 * (self.mass ** (1/3)))

    @property
    def angle(self) -> float:
        """Calculate and return the angle of the ball's velocity vector."""
        return float(math.degrees(math.atan2(self.velocity.y, self.velocity.x)))

    @property
    def speed(self) -> float:
        """Calculate and return the speed of the ball."""
        return float(vp.mag(self.velocity))

    @property
    def momentum(self) -> vp.vector:
        """Calculate and return the momentum vector of the ball."""
        return self.velocity * self.mass

    @property
    def momentum_mag(self) -> float:
        """Calculate and return the magnitude of the ball's momentum."""
        return float(vp.mag(self.momentum))

    @property
    def kinetic_energy(self) -> float:
        """Calculate and return the kinetic energy of the ball."""
        return 0.5 * self.mass * (self.speed**2)

    @classmethod
    def disable_gui(cls, no_gui: bool) -> None:
        """
        Enables or disables the GUI.

        Args:
            no_gui (bool): Flag to indicate where GUI should be disabled (True = disable GUI).
        """
        cls._no_gui = no_gui

    def update_position(self, dt: float) -> None:
        """
        Update the position of the ball based on its velocity and time step.

        Args:
            dt (float): Time step for the update.
        """
        self.position += self.velocity * dt

        if Ball._no_gui is False:
            self._sphere.pos = self.position
            self._label.pos = self.position
            if self._collision_blob and self.collision_point_offset:
                self._collision_blob.pos = self.position - self.collision_point_offset

    def mark_collision_point(self, cor: float) -> None:
        """
        Mark the collision point with a visible blob

        Args:
            cor (float): The Coefficient of Restitution
        """
        if Ball._no_gui is False:
            assert self.collision_point_offset

            # The scaling factor is arbitrary to look okay with CORs between 0-1
            scaling_factor: float = 1.5

            self._collision_blob = vp.ellipsoid(color=vp.color.yellow,
                                                axis=self.collision_point_offset,
                                                size=vp.vector(self.radius/scaling_factor,
                                                               self.radius*scaling_factor,
                                                               self.radius*scaling_factor) * cor)

    def set_visibility(self, is_visible: bool) -> None:
        """
        Sets the visibility of the Ball to visible or hidden

        Args:
            is_visible (bool): True=Ball is visible, False=Ball is hidden
        """
        if Ball._no_gui is False:
            self._sphere.visible = is_visible
            self._label.visible = is_visible
            self._sphere.make_trail = is_visible
            self._sphere.clear_trail()
