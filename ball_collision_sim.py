#!/usr/bin/env python
"""
Ball Collision Simulator

This module simulates the elastic, inelastic, or partially elastic collisions between two balls
using the VPython library. It contains classes to define the physical and visual properties of
the balls, perform the simulation, and visualize the collision in a 2D space. The module also
supports running the simulation without a graphical user interface (GUI).

Enums:
    - CollisionType: Indicates what type of collision to simulate: elastic, inelastic, or partially elastic.
    - Balls: Used to index the correct ball from the list of active balls
    - BallTrajectories: Indicates if the balls are converging, diverging, or at a constant
                        distance.

Classes:
    - PhysicsParameters: Data class to store the physical parameters of a ball.
    - BallParameters: Data class for storing the physical and visual properties of a ball.
    - SimParameters: Data class for storing the parameters for the BallCollisionSimulator
    - Ball: Represents a ball in the simulation, managing its state and visualization.
    - CollisionInfo: Data class for storing collision information.
    - IntersectionInfo: Data class for storing intersection information of ball paths.
    - SimulatorState: Data class for storing the state of the simulator
    - BallCollisionSimulator: Manages the entire simulation process, including initialization, 
                              running the simulation, and handling collisions.

Functions:
    - main: The main entry point of the program, running the simulation with either predefined 
      test parameters or user input.

Usage:
    Run this module as a script to start the simulation. The user can choose to input custom
    parameters for the balls and the simulation time or use predefined test parameters with 
    the `--test` argument. Additionally, the `--no_gui` argument can be used to run the simulation
    without the GUI.
"""
from __future__ import annotations
from typing import Tuple, List, Optional, Final
import argparse
from dataclasses import dataclass
import math
from copy import copy, deepcopy
from enum import Enum, IntEnum, auto
import vpython as vp
import readchar

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


class PhysicsParameters:
    """
    Class to store the physical parameters of a ball.
    """
    def __init__(self, mass: float, position: Tuple[float, float], velocity: Tuple[float, float]):
        """
        Initialize the PhysicsParameters object.

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
        collision_type (CollisionType): Type of collision to simulate:
                                        (elastic, inelastic, or partially elastic).
        cor (float): Coefficient of Restitution (used for partially elastic collisions)
    """

    ball_params: List[BallParameters]
    simulation_time: float
    collision_type: CollisionType
    cor: float


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
            params (BallParameters): Parameters for the ball.
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


@dataclass
class CollisionInfo:
    """
    Data class to store information about a collision.

    Attributes:
        time (float): Time of collision.
        ball1 (Ball): Ball 1 object (for elastic or partially elastic collisions)
        ball2 (Ball): Ball 2 object (for elastic or partially elastic  collisions)
        merged_ball (Ball): Merged ball (for inelastic collision)
    """
    time: float
    ball1: Ball
    ball2: Ball
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
class SimulatorState:
    """
    Data class to store the state of the Simulator.

    Attributes:
        balls (List[Ball]): List of Ball objects.  
                            - Index 0 = Ball 1.  
                            - Index 1 = Ball 2.  
                            - Index 2 = Merged Ball (Optional).  
        momentum (vp.vector): Total momentum of both balls (N⋅s).
        kinetic_energy (float): Total kinetic energy of both balls (J).
        relative_speed (float): Relative speed of the two balls with respect to each other (m/s).
        distance (float): Distance of the two balls (m).
        trajectories (BallTrajectories): What the balls current trajectories are:
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
        """Alias for Ball 1 (self.balls[Balls.BALL1])"""
        return self.balls[Balls.BALL1]

    @property
    def ball2(self) -> Ball:
        """Alias for Ball 2 (self.balls[Balls.BALL2])"""
        return self.balls[Balls.BALL2]


class BallCollisionSimulator:
    """
    Class to simulate the collision between two balls.

    Attributes:
        sim_params (SimParameters): Parameters given to the BallCollisionSimulator.
        collision_info (CollisionInfo): Information about the collision.
        intersect_info (IntersectionInfo): Information about the intersection.
        balls (List[Ball]): List of Ball objects.  
                            - Index 0 = Ball 1.  
                            - Index 1 = Ball 2.  
                            - Index 2 = Merged Ball (Optional).  
        initial (SimulatorState): Initial state of simulator before simulation is run.
    """

    # Flag to indicate whether the GUI should be disabled (True = no GUI)
    _no_gui = False

    SIMULATION_TIME_AFTER_COLLISION: Final[int] = 3
    """
    Defines how many seconds the simulation will run after collision.
    Note, if this time exceeds the total simulation time, the simulation will stop at
    the simulation time first.
    """

    LOOP_EXECUTION_RATE: Final[int] = 100
    """Defines how many times a second the simulation loop will execute."""

    DT: Final[float] = 0.01
    """Defines what the time delta between each simulation loop iteration is."""

    def __init__(self,
                 ball_params: List[BallParameters],
                 simulation_time: float,
                 collision_type: CollisionType = CollisionType.ELASTIC,
                 cor: Optional[float] = None):
        """
        Args:
            ball_params (List[BallParameters]): List of parameters for each ball.
            simulation_time (float): Total time to simulate.
            collision_type (CollisionType): Type of collision to simulate:
                                            (elastic, inelastic, or partially elastic).
            cor (Optional[float]): Coefficient of Restitution (used for partially elastic collisions)
        """
        self._scene: Optional[vp.canvas] = None

        # Error check given COR
        if cor is None or collision_type != CollisionType.PARTIAL:
            cor = 1.0
        elif cor <= 0.0 or cor >= 1.0:
            raise ValueError("COR must be > 0.0 and < 1.0.")

        self.sim_params: SimParameters = SimParameters(ball_params,
                                                       simulation_time,
                                                       collision_type,
                                                       cor)

        self.collision_info: Optional[CollisionInfo] = None
        self.intersect_info: Optional[IntersectionInfo] = None

        # Create scene and grid if GUI enabled
        if BallCollisionSimulator._no_gui is False:
            self._scene = vp.canvas(
                title=f'{self.sim_params.collision_type.name} Collision Simulator',
                width=800, height=800)
            
            if self.sim_params.collision_type == CollisionType.PARTIAL:
                self._scene.append_to_title(f', COR: {self.sim_params.cor}')

            # Set up grid
            self._create_grid_and_axes()

        # Create ball objects
        self.balls: List[Ball] = []
        for ball_param in ball_params:
            self.balls.append(Ball(ball_param))

        # Store Simulator State for later
        self.initial: SimulatorState = SimulatorState(deepcopy(self.balls),
                                                      self.momentum,
                                                      self.kinetic_energy,
                                                      self.relative_speed,
                                                      self.distance,
                                                      self.trajectories)

    def __del__(self) -> None:
        """
        Deletes the scene and sets the reference to None to allow scene to disappear from GUI
        """
        if self._scene:
            self._scene.delete()
            self._scene = None

    @property
    def relative_speed(self) -> float:
        """Relative speed of the two balls with respect to each other (m/s)."""
        if self.merged_ball:
            return 0.0
        else:
            return float(vp.mag(self.ball1.velocity - self.ball2.velocity))

    @property
    def distance(self) -> float:
        """Distance between the two balls (m)."""
        if self.merged_ball:
            return 0.0
        else:
            return float(vp.mag(self.ball1.position - self.ball2.position))

    @property
    def momentum(self) -> vp.vector:
        """Momentum of both balls (N⋅s)."""
        if self.merged_ball:
            return self.merged_ball.momentum
        else:
            return self.ball1.momentum + self.ball2.momentum

    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy of both balls together, before collision (J)."""
        if self.merged_ball:
            return self.merged_ball.kinetic_energy
        else:
            return self.ball1.kinetic_energy + self.ball2.kinetic_energy

    @property
    def trajectories(self) -> BallTrajectories:
        """What the balls' current trajectories are (constant, diverging, or converging)"""
        if self.merged_ball:
            return BallTrajectories.MERGED
        else:
            current_distance = vp.mag(self.ball1.position - self.ball2.position)

            # Project positions a tiny bit into the future
            future_pos1 = self.ball1.position + (self.ball1.velocity * self.DT)
            future_pos2 = self.ball2.position + (self.ball2.velocity * self.DT)
            future_distance = vp.mag(future_pos1 - future_pos2)

            # Compare future distance to current distance
            distance_change = round(future_distance - current_distance, ndigits=6)

            if distance_change > 0.0:
                return BallTrajectories.DIVERGING
            elif distance_change < 0.0:
                return BallTrajectories.CONVERGING
            else:
                return BallTrajectories.CONSTANT

    @property
    def ke_lost(self) -> float:
        """The amount of KE lost since start of simulation (J)."""
        if self.merged_ball:
            return self.initial.kinetic_energy - self.merged_ball.kinetic_energy
        else:
            return self.initial.kinetic_energy - self.kinetic_energy

    @property
    def ball1(self) -> Ball:
        """Alias for Ball 1 (self.balls[Balls.BALL1])"""
        return self.balls[Balls.BALL1]

    @property
    def ball2(self) -> Ball:
        """Alias for Ball 2 (self.balls[Balls.BALL2])"""
        return self.balls[Balls.BALL2]

    @property
    def merged_ball(self) -> Optional[Ball]:
        """Alias for Merged Ball (self.balls[Balls.MERGED])"""
        if len(self.balls) > Balls.MERGED:
            return self.balls[Balls.MERGED]
        else:
            return None

    @classmethod
    def disable_gui(cls, no_gui: bool) -> None:
        """
        Enables or disables the GUI.

        Args:
            no_gui (bool): Flag to indicate where GUI should be disabled (True = disable GUI).
        """
        cls._no_gui = no_gui
        Ball.disable_gui(no_gui)

    @classmethod
    def create_simulator(cls,
                         phys_params: List[PhysicsParameters],
                         simulation_time: float,
                         collision_type: CollisionType = CollisionType.ELASTIC,
                         cor: Optional[float] = None) \
            -> BallCollisionSimulator:
        """
        Create a BallCollisionSimulator instance with given parameters.

        Args:
            phys_params (List[PhysicsParameters]): List of physics parameters for each ball.
            simulation_time (float): Total time to simulate.
            collision_type (CollisionType): Type of collision to simulate:
                                            (elastic, inelastic, or partially elastic).
            cor (Optional[float]): Coefficient of Restitution (used for partially elastic collisions)

        Returns:
            BallCollisionSimulator: An instance of the simulator.
        """
        ball_params: List[BallParameters] = []
        ball_params.append(BallParameters(phys_params[0], color=vp.color.blue, name='1'))
        ball_params.append(BallParameters(phys_params[1], color=vp.color.red, name='2'))

        return cls(ball_params, simulation_time, collision_type, cor)

    @staticmethod
    def quit_simulation() -> None:
        """Stops the VPython server."""
        if BallCollisionSimulator._no_gui is False:
            # We don't import vp_services until needed, because importing it will start
            # the server, if not started already.
            import vpython.no_notebook as vp_services  # type: ignore[import-untyped]
            vp_services.stop_server()

    def _create_grid_and_axes(self) -> None:
        """Create a grid and axes for the simulation scene."""
        grid_range: int = 10
        step: int = 1

        for x in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(pos=[vp.vector(x, -grid_range, 0), vp.vector(x, grid_range, 0)],
                     color=vp.color.gray(0.5) if x != 0 else vp.color.white)
        for y in vp.arange(-grid_range, grid_range + step, step):
            vp.curve(pos=[vp.vector(-grid_range, y, 0), vp.vector(grid_range, y, 0)],
                     color=vp.color.gray(0.5) if y != 0 else vp.color.white)

        # Create axis labels
        vp.label(pos=vp.vector(grid_range + 0.5, 0, 0), text='X', height=16, box=False)
        vp.label(pos=vp.vector(0, grid_range + 0.5, 0), text='Y', height=16, box=False)

    @staticmethod
    def _print_ball_state(balls: List[Ball]) -> None:
        """
        Print detailed information about the state of the balls listed.

        Args:
            balls (List[Ball]): List of Ball objects.
        """
        print()
        for ball in balls:
            print(f'Ball {ball.name}:')
            print(f'  Mass: {ball.mass} kg')
            print(f'  Radius: {ball.radius:.3g} m')
            print(f'  Position: ({ball.position.x:.3g}, {ball.position.y:.3g})')
            print(f'  Velocity: ({ball.velocity.x:.5g}, {ball.velocity.y:.5g}), or {
                ball.speed:.3g} m/s at {ball.angle:.3g}°')
            print(f'  Momentum: ({ball.momentum.x:.3g}, {ball.momentum.y:.3g}), or {
                ball.momentum_mag:.3g} N⋅s at {ball.angle:.3g}°')
            print(f'  Kinetic Energy: {ball.kinetic_energy:.3g} J')
        print()

    def _calculate_intersection(self) -> None:
        """
        Calculate the intersection point of the paths of the two balls.

        This method determines if and where the paths of two balls intersect
        within the simulation time. It uses a parametric approach to find
        the intersection of two line segments representing the ball paths.

        The algorithm uses parametric variables t and u (range 0 to 1) which 
        represent positions along ball1 and ball2's paths respectively. An 
        intersection exists if both t and u are between 0 and 1.

        Notes on 't' and 'u':  
        - 't' and 'u' are parametric variables. They represent a position along  
           each line segment, scaled from 0 to 1.  
        - For 't' (related to ball1's path):  
            - 't = 0' means you're at ball1's starting point (x1, y1)  
            - 't = 1' means you're at ball1's ending point (x1_end, y1_end)  
            - '0 < t < 1' means you're somewhere along ball1's path  
        - 'u' works the same way but for ball2's path.  
        -  If '0 <= t <= 1' and '0 <= u <= 1', it means the intersection point lies  
           within both line segments, i.e., the balls' paths genuinely intersect.


           The method stores the intersection information in self.intersect_info
        if an intersection is found within the valid range of both paths.

        Returns:
            None
        """
        # Extract initial positions and velocities
        x1, y1 = self.initial.ball1.position.x, self.initial.ball1.position.y
        x2, y2 = self.initial.ball2.position.x, self.initial.ball2.position.y
        vx1, vy1 = self.initial.ball1.velocity.x, self.initial.ball1.velocity.y
        vx2, vy2 = self.initial.ball2.velocity.x, self.initial.ball2.velocity.y

        # Calculate end points of the line segments
        x1_end = x1 + vx1 * self.sim_params.simulation_time
        y1_end = y1 + vy1 * self.sim_params.simulation_time
        x2_end = x2 + vx2 * self.sim_params.simulation_time
        y2_end = y2 + vy2 * self.sim_params.simulation_time

        # Calculate the intersection of these line segments
        denominator = (x1 - x1_end) * (y2 - y2_end) - (y1 - y1_end) * (x2 - x2_end)

        # If lines are parallel, return None
        if denominator == 0:
            return None

        # Calculate parameters t and u
        # t represents how far along the first ball's path the intersection occurs
        # u represents the same for the second ball's path
        t = ((x1 - x2) * (y2 - y2_end) - (y1 - y2) * (x2 - x2_end)) / denominator
        u = -((x1 - x1_end) * (y1 - y2) - (y1 - y1_end) * (x1 - x2)) / denominator

        # Correct -0.0 to 0.0 for consistency  (float-point math anomalies)
        t = 0.0 if t == -0.0 else t
        u = 0.0 if u == -0.0 else u

        # Check if intersection point is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix = x1 + t * (x1_end - x1)
            iy = y1 + t * (y1_end - y1)

            # Store intersection info
            self.intersect_info = IntersectionInfo(
                position=vp.vector(ix, iy, 0),
                ball1_time=t * self.sim_params.simulation_time,
                ball2_time=u * self.sim_params.simulation_time
            )

    def _calculate_collision_point(self) -> None:
        """Calculate the single point where the balls touch during collision."""
        x1: vp.vector = self.ball1.position
        x2: vp.vector = self.ball2.position
        r1: float = self.ball1.radius
        r2: float = self.ball2.radius

        # Vector from ball1 to ball2
        direction: vp.vector = x2 - x1
        distance: float = direction.mag

        # Normalize the direction vector
        if distance != 0:
            direction = direction.norm()
        else:
            # If balls are at the same position, use velocity difference as direction
            direction = (self.ball2.velocity - self.ball1.velocity).norm()

        # Calculate the collision point
        # This point is r1 / (r1 + r2) of the way from ball1 to ball2
        collision_point: vp.vector = x1 + direction * (distance * r1 / (r1 + r2))
        self.ball1.collision_point = self.ball2.collision_point = collision_point

        # Calculate collision point offsets
        self.ball1.collision_point_offset = self.ball1.position - collision_point
        self.ball2.collision_point_offset = self.ball2.position - collision_point

        # Mark the collision point on the balls
        self.ball1.mark_collision_point(self.sim_params.cor)
        self.ball2.mark_collision_point(self.sim_params.cor)

    
    def _elastic_collision_physics(self) -> None:
        """
        Calculate and update ball velocities after an elastic or partially elastic collision.

        This method simulates an elastic collision between two balls by:
        1. Computing the collision normal vector (direction of impact)
        2. Decomposing velocities into components parallel and perpendicular to this normal
        3. Applying collision physics equations to compute new velocities
        4. Reconstructing velocity vectors and updating ball states

        The collision normal vector is crucial as it defines the line along which 
        momentum and energy are exchanged during collision. It typically points 
        from one ball's center to the other's. Components of velocity along this 
        normal are modified by the collision, while perpendicular components 
        (tangential) remain unchanged.

        Special cases are handled when balls occupy the same position, where
        velocity vectors are used to infer a meaningful collision normal.
        """
        def _check_for_neg_zero() -> None:
            """If any of the x,y components have -0 in them, change them to 0"""
            self.ball1.velocity.x = 0.0 if self.ball1.velocity.x == -0 else self.ball1.velocity.x
            self.ball1.velocity.y = 0.0 if self.ball1.velocity.y == -0 else self.ball1.velocity.y
            self.ball2.velocity.x = 0.0 if self.ball2.velocity.x == -0 else self.ball2.velocity.x
            self.ball2.velocity.y = 0.0 if self.ball2.velocity.y == -0 else self.ball2.velocity.y

        m1: float = self.ball1.mass
        m2: float = self.ball2.mass
        v1: vp.vector = self.ball1.velocity
        v2: vp.vector = self.ball2.velocity
        x1: vp.vector = self.ball1.position
        x2: vp.vector = self.ball2.position

        # Only mark collision points on partially elastic collisions
        if self.sim_params.collision_type == CollisionType.PARTIAL:
            self._calculate_collision_point()

        # Calculate the normal vector of collision
        diff: vp.vector = x1 - x2
        normal: vp.vector

        # Handle special cases for normal vector calculation
        # Balls are superimposed (can occur due to user-defined initial conditions)
        if diff.mag == 0:  # If balls are in the same position
            if v1 == v2:  # If balls have the same velocity
                # Use either v1 or v2, they're the same
                normal = v1.norm()
            else:
                # Use the difference of velocities as the normal direction
                normal = (v1 - v2).norm()
        else:
            # Normal case: normal is the unit vector from ball2 to ball1
            normal = diff.norm()

        # Calculate the tangential vector (perpendicular to normal)
        tangent: vp.vector = vp.vector(-normal.y, normal.x, 0)

        # Project velocities onto normal and tangential vectors
        v1n: float = v1.dot(normal)   # Normal component of v1
        v1t: float = v1.dot(tangent)  # Tangential component of v1
        v2n: float = v2.dot(normal)   # Normal component of v2
        v2t: float = v2.dot(tangent)  # Tangential component of v2

        # Calculate new normal velocities using elastic collision formula
        v1n_new: float = ((self.sim_params.cor * m2 * (v2n - v1n)) + (m1 * v1n) + (m2 * v2n)) / (m1 + m2)
        v2n_new: float = ((self.sim_params.cor * m1 * (v1n - v2n)) + (m1 * v1n) + (m2 * v2n)) / (m1 + m2)

        # Reconstruct the new velocity vectors
        # New velocity = (new normal component * normal vector) +
        #                (unchanged tangential component * tangential vector)
        self.ball1.velocity = (v1n_new * normal) + (v1t * tangent)
        self.ball2.velocity = (v2n_new * normal) + (v2t * tangent)

        # Check if any of the velocities have any -0's to get rid of
        _check_for_neg_zero()

    def _inelastic_collision_physics(self) -> None:
        """Calculate and update the physics of the balls after inelastic collision."""
        # Calculate new mass
        total_mass = self.ball1.mass + self.ball2.mass

        # New velocity (conservation of momentum)
        new_velocity = (self.ball1.momentum + self.ball2.momentum) / total_mass

        # New position (center of mass)
        new_position = ((self.ball1.position * self.ball1.mass) +
                        (self.ball2.position * self.ball2.mass)) / total_mass

        # Hide original balls
        self.ball1.set_visibility(is_visible=False)
        self.ball2.set_visibility(is_visible=False)

        # New merged ball parameters
        merged_params = BallParameters(
            PhysicsParameters(total_mass,
                              (new_position.x, new_position.y),
                              (new_velocity.x, new_velocity.y)),
            color=vp.color.green,
            name='1-2'
        )

        # Create merged ball
        self.balls.append(Ball(merged_params))

    def _process_post_collision_physics(self) -> None:
        """Calculate and update the physics of the balls after collision."""
        if self.sim_params.collision_type in [CollisionType.ELASTIC, CollisionType.PARTIAL]:
            self._elastic_collision_physics()
        # Else, inelastic collision
        else:
            self._inelastic_collision_physics()

    def _verify_conservation_of_momentum(self) -> None:
        """Verify that momentum is conserved after the collision."""
        # Verify momentum has been conserved
        assert round(self.initial.momentum.mag, ndigits=3) == \
            round(self.momentum.mag, ndigits=3), \
            f'Initial total: {self.initial.momentum.mag}, Final total: {self.momentum.mag}'

    def _verify_conservation_of_ke(self) -> None:
        """
        Verify that kinetic energy is conserved after the collision for elastic collisions,
        or calculate how much kinetic energy was lost after the collision for partially
        elastic or inelastic collisions.
        """
        # if the balls haven't merged, then check their kinetic energy
        if self.sim_params.collision_type == CollisionType.ELASTIC:
            # Verify KE has been conserved
            assert round(self.ke_lost, ndigits=3) == 0.0, \
                f'Initial total: {self.initial.kinetic_energy}, Final total: {self.kinetic_energy}'
        # Else, check the loss of KE from inelastic or partially elastic collision
        else:
            print(f'Kinetic Energy lost in collision: {self.ke_lost:.3g} J')

    def _run_simulation(self) -> None:
        """Run the simulation loop."""
        time_elapsed: float = 0.0
        last_trajectory: Optional[BallTrajectories] = None

        def print_current_trajectory() -> None:
            nonlocal last_trajectory
            current_trajectory: BallTrajectories = self.trajectories
            if last_trajectory != current_trajectory:
                if current_trajectory == BallTrajectories.CONVERGING:
                    print(f't={time_elapsed:.3g}s: Balls are Converging at: {self.relative_speed:.3g} m/s')
                elif current_trajectory == BallTrajectories.DIVERGING:
                    print(f't={time_elapsed:.3g}s: Balls are Diverging at: {self.relative_speed:.3g} m/s')
                elif current_trajectory == BallTrajectories.MERGED:
                    print(f't={time_elapsed:.3g}s: Balls have Merged.')
                else:
                    print(f't={time_elapsed:.3g}s: Balls are maintaining a constant distance. Relative speed: {
                        self.relative_speed:.3g} m/s')
                last_trajectory = current_trajectory

        while True:
            vp.rate(self.LOOP_EXECUTION_RATE)
            print_current_trajectory()

            # If a collision hasn't occured already and the ball's positions are within the
            # distance of both radiuses, we have a collision
            if not self.collision_info and \
                    vp.mag(self.ball1.position - self.ball2.position) <= \
                    (self.ball1.radius + self.ball2.radius):
                # update balls based on physics of collision
                self._process_post_collision_physics()

                print_current_trajectory()

                # Store collision state info for later
                self.collision_info = CollisionInfo(ball1=copy(self.ball1),
                                                    ball2=copy(self.ball2),
                                                    merged_ball=copy(self.merged_ball),
                                                    time=time_elapsed)

            # If we've had a collision, check if we've reached the "run a bit after the collision"
            # time. If we haven't had a collision, check if simulation duration has past
            if (self.collision_info and time_elapsed >
                (self.collision_info.time + self.SIMULATION_TIME_AFTER_COLLISION)) \
                    or (time_elapsed > self.sim_params.simulation_time):
                break

            if self.merged_ball:  # If we have a merged ball after collision
                self.merged_ball.update_position(self.DT)
            else:
                self.ball1.update_position(self.DT)
                self.ball2.update_position(self.DT)

            time_elapsed += self.DT

    def run(self) -> None:
        """
        Run the simulation.
        """
        print('\n***************************************************')
        print('Initial Conditions:')
        self._print_ball_state([self.initial.ball1, self.initial.ball2])
        print(f'Initial Distance: {self.initial.distance:.3g} m')
        print(f'Sum of Radii: {(self.ball1.radius + self.ball2.radius):.3g}')
        print(f'Total Momentum: ({self.initial.momentum.x:.3g}, {
            self.initial.momentum.y:.3g}), or {
            vp.mag(self.initial.momentum):.3g} N⋅s at {
            math.degrees(math.atan2(self.initial.momentum.y, self.initial.momentum.x)):.3g}°')
        print(f'Total Kinetic Energy: {(self.initial.kinetic_energy):.3g} J')
        print()

        # Run the simulation
        self._run_simulation()

        print()

        # If collision occured
        if self.collision_info:
            print(f'Collision occured at time: {self.collision_info.time:.3g} secs')
            print('\nPost Collision Conditions:')
            if self.collision_info.merged_ball:
                self._print_ball_state([self.collision_info.merged_ball])
            else:
                self._print_ball_state([self.collision_info.ball1, self.collision_info.ball2])
        # Else no collision, see if the paths intersected
        else:
            print(f'No collision occured during simulation time of {
                self.sim_params.simulation_time} secs.')

            # Calculate path intersection (if any)
            self._calculate_intersection()
            if self.intersect_info:
                print('Paths did intersect though:')
                print(f'  Path Intersection Point: ({self.intersect_info.position.x:.3g}, {
                    self.intersect_info.position.y:.3g})')
                print(f'  Time for Ball 1 to reach intersection: {
                    self.intersect_info.ball1_time:.3g} secs')
                print(f'  Time for Ball 2 to reach intersection: {
                    self.intersect_info.ball2_time:.3g} secs')
            else:
                print('No path intersection found either.')

        self._verify_conservation_of_momentum()
        self._verify_conservation_of_ke()

        print('###################################################')


def main() -> None:
    """
    Main entry point for the Ball Collision Simulator.

    This function parses command-line arguments to determine whether to use predefined 
    test parameters or prompt the user for input. It initializes the simulator with the
    appropriate parameters and runs the simulation. If the `--no_gui` flag is set, the
    simulation runs without a graphical user interface (GUI).

    * Command-line Arguments:  
        `--test`: Run the simulation with predefined test parameters.  
        `--no_gui`: Run the simulation without the GUI.  

    * Prompts:  
        - If not using predefined test parameters, the user is prompted to enter:  
            - Simulation type: (elastic, inelastic, or partially elastic)  
            - Coefficient of Restitution (COR): (0.0 < COR < 1.0)  
            - Mass (kg) for Ball 1 and Ball 2  
            - Initial position (x, y) in meters for Ball 1 and Ball 2  
            - Initial velocity (vx, vy) in m/s for Ball 1 and Ball 2  
            - Simulation time in seconds  

    * What it does:  
        - Prints simulation details to the console.  
        - Runs the simulation, optionally displaying it in a VPython GUI window.  
        - Waits for a keypress to exit if the GUI is enabled.  

    """
    def _get_user_input() -> Tuple[List[PhysicsParameters], float, CollisionType, float]:
        """
        Get user input for ball parameters, simulation time, and collision type.

        Returns:
            Tuple[List[PhysicsParameters], float, CollisionType, Optional[float]]:
            Parameters for both balls, simulation time, and collision type (elastic,
            inelastic, or partially elastic), and optionally a Coefficient of Restitution (COR).
        """
        def get_float(prompt: str) -> float:
            while True:
                try:
                    return float(input(prompt))
                except ValueError:
                    print('Please enter a valid number.')

        def get_vector(prompt: str) -> Tuple[float, float]:
            while True:
                try:
                    x, y = map(float, input(prompt).split(','))
                    return (x, y)
                except ValueError:
                    print('Please enter two numbers separated by a comma.')

        def get_collision_type() -> CollisionType:
            collision_selection = None
            while collision_selection not in ['e', 'i', 'p']:
                collision_selection = input(
                    "Enter collision type ('e'=elastic, 'i'=inelastic), 'p'= partial: ").lower()

            if collision_selection == 'i':
                return CollisionType.INELASTIC
            elif collision_selection == 'p':
                return CollisionType.PARTIAL
            else:
                return CollisionType.ELASTIC

        def get_cor(prompt: str) -> float:
            while True:
                try:
                    cor: float = float(input(prompt))
                    if cor > 0 and cor < 1:
                        return cor
                    else:
                        print('COR must be > 0.0 and < 1.0.')
                except ValueError:
                    print('Please enter a valid number.')

        collision_type = get_collision_type()

        cor: float = 1.0
        if collision_type == CollisionType.PARTIAL:
            cor = get_cor("\nEnter Coefficient of Restitution (0.0 < cor < 1.0): ")

        print("Enter parameters for Ball 1:")
        mass1: float = get_float("Mass (kg): ")
        position1: Tuple[float, float] = get_vector("Position (x,y) in meters: ")
        velocity1: Tuple[float, float] = get_vector("Velocity (x,y) in m/s: ")

        print("\nEnter parameters for Ball 2:")
        mass2: float = get_float("Mass (kg): ")
        position2: Tuple[float, float] = get_vector("Position (x,y) in meters: ")
        velocity2: Tuple[float, float] = get_vector("Velocity (x,y) in m/s: ")

        simulation_time: float = get_float("\nEnter simulation time (seconds): ")

        return ([PhysicsParameters(mass1, position1, velocity1),
                 PhysicsParameters(mass2, position2, velocity2)],
                simulation_time,
                collision_type,
                cor)

    parser = argparse.ArgumentParser(description='Ball Collision Simulator')
    parser.add_argument('--test', action='store_true', help='Run with pre-defined test case')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    if args.no_gui is True:
        BallCollisionSimulator.disable_gui(True)

    if args.test:
        # Pre-defined test case
        ball_params: List[PhysicsParameters] = [PhysicsParameters(mass=1, position=(2.4, 2), velocity=(-1, -1)),
                                                PhysicsParameters(mass=3, position=(-2, -2), velocity=(1, 1))]

        simulation_time: float = 10.0  # secs

        # collision_type = CollisionType.ELASTIC
        # collision_type = CollisionType.INELASTIC
        collision_type = CollisionType.PARTIAL

        cor: float = 0.5
    else:
        # Get user input
        ball_params, simulation_time, collision_type, cor = _get_user_input()

    ball_collision_sim: BallCollisionSimulator = BallCollisionSimulator.create_simulator(ball_params,
                                                                                         simulation_time,
                                                                                         collision_type,
                                                                                         cor)

    ball_collision_sim.run()

    if args.no_gui is False:
        print("Press any key to exit...")
        readchar.readkey()
        BallCollisionSimulator.quit_simulation()


if __name__ == '__main__':
    main()
