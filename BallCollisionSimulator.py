"""
Ball Collision Simulator

This module simulates the elastic or inelastic collision between two balls using the VPython
library. It contains classes to define the physical and visual properties of the balls, perform
the simulation, and visualize the collision in a 2D space. The module also supports running
the simulation without a graphical user interface (GUI).

Enums:
    - CollisionType: Indicates what type of collision to simulate, elastic or inelastic

Classes:
    - PhysicsParameters: Stores the physical parameters of a ball.
    - BallParameters: Combines physical and visual properties of a ball.
    - Ball: Represents a ball in the simulation, managing its state and visualization.
    - CollisionInfo: Data class for storing collision information.
    - IntersectionInfo: Data class for storing intersection information of ball paths.
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
from typing import Tuple, List, Optional
import argparse
from dataclasses import dataclass
from copy import copy
import vpython as vp
import readchar
from enum import Enum


class CollisionType(Enum):
    """
    Enum to indicate the type of Collision, elastic or inelastic.
    """
    ELASTIC = 1
    INELASTIC = 2


class PhysicsParameters:
    """
    Class to store the physical parameters of a ball.

    Attributes:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Initial position of the ball (x, y) in meters.
        velocity (Tuple[float, float]): Initial velocity of the ball (vx, vy) in m/s.
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


class BallParameters:
    """
    Class to store all parameters of a ball, including physics and visual properties.

    Attributes:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Initial position of the ball (x, y) in meters.
        velocity (Tuple[float, float]): Initial velocity of the ball (vx, vy) in m/s.
        color (vp.color, optional): Color of the ball. Defaults to red.
        name (str, optional): Name or identifier for the ball. Defaults to ''.
    """

    def __init__(self, physics: PhysicsParameters, color: vp.color = vp.color.red, name: str = ''):
        """
        Args:
            physics (PhysicsParameters): Physical parameters of the ball.
            color (vp.color, optional): Color of the ball. Defaults to red.
            name (str, optional): Name or identifier for the ball. Defaults to ''.
        """
        self.mass: float = physics.mass
        self.position: vp.vector = physics.position
        self.velocity: vp.vector = physics.velocity
        self.color: vp.color = color
        self.name: str = name


class Ball:
    """
    Class representing a ball in the simulation.

    Attributes:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Position of the ball (x, y) (m).
        velocity (Tuple[float, float]): Velocity of the ball (vx, vy) (m/s).
        radius (float): Radius of the ball (m).
        name (str): Name of the Ball.
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
        self.mass: float = params.mass
        self.position: vp.vector = params.position
        self.velocity: vp.vector = params.velocity

        # Radius proportional to the mass (0.5m for 1 kg)
        self.radius: float = 0.5 * (self.mass ** (1/3))

        self.name: str = params.name

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

    @property
    def angle(self) -> float:
        """Calculate and return the angle of the ball's velocity vector."""
        return vp.degrees(vp.atan2(self.velocity.y, self.velocity.x))

    @property
    def speed(self) -> float:
        """Calculate and return the speed of the ball."""
        return vp.mag(self.velocity)

    @property
    def momentum(self) -> vp.vector:
        """Calculate and return the momentum vector of the ball."""
        return self.velocity * self.mass

    @property
    def momentum_mag(self) -> float:
        """Calculate and return the magnitude of the ball's momentum."""
        return vp.mag(self.momentum)

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
        ball1 (Ball): Ball 1 object (for elastic collision)
        ball2 (Ball): Ball 2 object (for elastic collision)
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


class BallCollisionSimulator:
    """
    Class to simulate the collision between two balls.

    Attributes:
        ball1_params (BallParameters): Parameters for Ball 1.
        ball2_params (BallParameters): Parameters for Ball 2.
        simulation_time (float): Amount of time to run the simulation (s).
        ball1 (Ball): Ball 1 object.
        ball2 (Ball): Ball 2 object.
        ball1_state_t0 (Ball): Ball 1 initial state.
        ball2_state_t0 (Ball): Ball 2 initial state.
        merged_ball (Ball): For inelastic collisions, this is the merged ball after collision.
        initial_distance (float): Initial distance of the two balls (m).
        total_momentum (vp.vector): Total momentum of both balls together (N⋅s).
        init_total_ke (float): Total kinetic energy of both balls together, before collision (J).
        final_total_ke (float): Total kinetic energy of both balls together, after collision (J).
        ke_lost (float): The amount of KE lost during collision (J)
        dot_product (float): Dot product of the two balls.
        relative_speed (float): Relative speed of the two balls with respect to each other (m/s).
        collision_info (CollisionInfo): Information about the collision.
        intersection_info (IntersectionInfo): Information about the intersection.
    """

    # Flag to indicate whether the GUI should be disabled (True = no GUI)
    _no_gui = False

    def __init__(self, ball1_params: BallParameters,
                 ball2_params: BallParameters,
                 simulation_time: float,
                 collision_type: CollisionType = CollisionType.ELASTIC):
        """
        Args:
            ball1_params (BallParameters): Parameters for the first ball.
            ball2_params (BallParameters): Parameters for the second ball.
            simulation_time (float): Total time to simulate.
            collision_type (CollisionType): Type of collision to simulate (elastic or inelastic).
        """
        self.ball1_params: BallParameters = ball1_params
        self.ball2_params: BallParameters = ball2_params
        self.simulation_time: float = simulation_time
        self.collision_type: CollisionType = collision_type

        self._scene: Optional[vp.canvas] = None
        self.ball1: Optional[Ball] = None
        self.ball2: Optional[Ball] = None
        self.ball1_state_t0: Optional[Ball] = None
        self.ball2_state_t0: Optional[Ball] = None
        self.merged_ball: Optional[Ball] = None
        self.initial_distance: Optional[float] = None
        self.total_momentum: Optional[vp.vector] = None
        self.init_total_ke: Optional[float] = None
        self.final_total_ke: Optional[float] = None
        self.ke_lost: Optional[float] = None
        self.dot_product: Optional[float] = None
        self.relative_speed: Optional[float] = None
        self.collision_info: Optional[CollisionInfo] = None
        self.intersection_info: Optional[IntersectionInfo] = None

    def __del__(self):
        """
        Deletes the scene and sets the reference to None to allow scene to disappear from GUI
        """
        if self._scene:
            self._scene.delete()
            self._scene = None

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
                         phys1_params: PhysicsParameters,
                         phys2_params: PhysicsParameters,
                         simulation_time: float,
                         collision_type: CollisionType = CollisionType.ELASTIC) \
                         -> BallCollisionSimulator:
        """
        Create a BallCollisionSimulator instance with given parameters.

        Args:
            phys1_params (PhysicsParameters): Physics parameters for the first ball.
            phys2_params (PhysicsParameters): Physics parameters for the second ball.
            simulation_time (float): Total time to simulate.
            collision_type (CollisionType): Type of collision to simulate (elastic or inelastic).

        Returns:
            BallCollisionSimulator: An instance of the simulator.
        """
        ball1_params: BallParameters = BallParameters(phys1_params, color=vp.color.blue, name='1')
        ball2_params: BallParameters = BallParameters(phys2_params, color=vp.color.red, name='2')

        return cls(ball1_params, ball2_params, simulation_time, collision_type)

    @staticmethod
    def quit_simulation() -> None:
        """Stop the VPython server."""
        if BallCollisionSimulator._no_gui is False:
            # We don't import vp_services until needed, because importing it will start
            # the server, if not started already.
            import vpython.no_notebook as vp_services
            vp_services.stop_server()

    def _init_simulation(self) -> None:
        """Initialize the simulation by setting up the scene and balls."""
        if BallCollisionSimulator._no_gui is False:
            self._scene = vp.canvas(title=f'{self.collision_type.name} Collision Simulation',
                                    width=800, height=800,
                                    center=vp.vector(0, 0, 0),
                                    background=vp.color.black)

            # Set up grid
            self._create_grid_and_axes()

        # Create ball objects
        self.ball1 = Ball(self.ball1_params)
        self.ball2 = Ball(self.ball2_params)

        # Store intial state for later
        self.ball1_state_t0 = copy(self.ball1)
        self.ball2_state_t0 = copy(self.ball2)

        # Calculate initial momentum
        self.total_momentum = self.ball1.momentum + self.ball2.momentum

        # Calculate initial kinetic energy
        self.init_total_ke = self.ball1.kinetic_energy + self.ball2.kinetic_energy

        # Calculate initial distance
        self.initial_distance = vp.mag(self.ball1.position - self.ball2.position)

        # Calculate the relative speed of the balls to each other
        self.dot_product = vp.dot(self.ball1.velocity - self.ball2.velocity,
                                  self.ball1.position - self.ball2.position)
        self.relative_speed = (self.ball1.velocity - self.ball2.velocity).mag

    def _create_grid_and_axes(self) -> None:
        """Create a grid and axes for the simulation scene."""
        GRID_RANGE: int = 10
        step: int = 1

        for x in vp.arange(-GRID_RANGE, GRID_RANGE + step, step):
            vp.curve(pos=[vp.vector(x, -GRID_RANGE, 0),
                          vp.vector(x, GRID_RANGE, 0)],
                     color=vp.color.gray(0.5) if x != 0 else vp.color.yellow)
        for y in vp.arange(-GRID_RANGE, GRID_RANGE + step, step):
            vp.curve(pos=[vp.vector(-GRID_RANGE, y, 0),
                          vp.vector(GRID_RANGE, y, 0)],
                     color=vp.color.gray(0.5) if y != 0 else vp.color.yellow)

        # Create axis labels
        vp.label(pos=vp.vector(GRID_RANGE + 0.5, 0, 0),
                 text='X', height=16, box=False)
        vp.label(pos=vp.vector(0, GRID_RANGE + 0.5, 0),
                 text='Y', height=16, box=False)

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
            print(f'  Radius: {ball.radius:.2f} m')
            print(f'  Position: ({ball.position.x:.2f}, {
                  ball.position.y:.2f})')
            print(f'  Velocity: ({ball.velocity.x:.2f}, {ball.velocity.y:.2f}), or {
                ball.speed:.2f} m/s at {ball.angle:.2f}°')
            print(f'  Momentum: ({ball.momentum.x:.2f}, {ball.momentum.y:.2f}), or {
                ball.momentum_mag:.2f} N⋅s at {ball.angle:.2f}°')
            print(f'  Kinetic Energy: {ball.kinetic_energy:.2f} J')
        print()

    def _calculate_intersection(self) -> None:
        """Calculate the intersection point of the paths of the two balls."""
        assert self.ball1_state_t0
        assert self.ball2_state_t0

        # Extract initial positions and velocities
        x1, y1 = self.ball1_state_t0.position.x, self.ball1_state_t0.position.y
        x2, y2 = self.ball2_state_t0.position.x, self.ball2_state_t0.position.y
        vx1, vy1 = self.ball1_state_t0.velocity.x, self.ball1_state_t0.velocity.y
        vx2, vy2 = self.ball2_state_t0.velocity.x, self.ball2_state_t0.velocity.y

        # Calculate end points of the line segments
        x1_end = x1 + vx1 * self.simulation_time
        y1_end = y1 + vy1 * self.simulation_time
        x2_end = x2 + vx2 * self.simulation_time
        y2_end = y2 + vy2 * self.simulation_time

        # Calculate the intersection of these line segments
        denominator = (x1 - x1_end) * (y2 - y2_end) - \
            (y1 - y1_end) * (x2 - x2_end)

        if denominator == 0:
            return None  # Lines are parallel

        t = ((x1 - x2) * (y2 - y2_end) - (y1 - y2)
             * (x2 - x2_end)) / denominator
        u = -((x1 - x1_end) * (y1 - y2) -
              (y1 - y1_end) * (x1 - x2)) / denominator

        # Correct -0.0 to 0.0
        if t == -0.0:
            t = 0.0
        if u == -0.0:
            u = 0.0

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            ix = x1 + t * (x1_end - x1)
            iy = y1 + t * (y1_end - y1)

            # Store intersection info
            self.intersection_info = IntersectionInfo(position=vp.vector(ix, iy, 0),
                                                      ball1_time=t * self.simulation_time,
                                                      ball2_time=u * self.simulation_time)

    def _elastic_collision_physics(self) -> None:
        """Calculate and update the physics of the balls after elastic collision."""
        def _check_for_neg_zero() -> None:
            """If any of the x,y components have -0 in them, change them to 0"""
            assert self.ball1
            assert self.ball2
            self.ball1.velocity.x = 0.0 if self.ball1.velocity.x == -0 else self.ball1.velocity.x
            self.ball1.velocity.y = 0.0 if self.ball1.velocity.y == -0 else self.ball1.velocity.y
            self.ball2.velocity.x = 0.0 if self.ball2.velocity.x == -0 else self.ball2.velocity.x
            self.ball2.velocity.y = 0.0 if self.ball2.velocity.y == -0 else self.ball2.velocity.y

        assert self.ball1
        assert self.ball2

        m1: float = self.ball1.mass
        m2: float = self.ball2.mass
        v1: vp.vector = self.ball1.velocity
        v2: vp.vector = self.ball2.velocity
        x1: vp.vector = self.ball1.position
        x2: vp.vector = self.ball2.position

        # Calculate the normal vector of collision
        diff: vp.vector = x1 - x2
        normal: vp.vector = None
        if diff.mag == 0:  # Balls are in the same position
            normal = (v1 - v2).norm()
        else:
            normal = diff.norm()

        # Calculate the tangential vector
        tangent: vp.vector = vp.vector(-normal.y, normal.x, 0)

        # Project velocities onto normal and tangential vectors
        v1n: float = v1.dot(normal)
        v1t: float = v1.dot(tangent)
        v2n: float = v2.dot(normal)
        v2t: float = v2.dot(tangent)

        # Calculate new normal velocities
        v1n_new: float = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
        v2n_new: float = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)

        # Convert scalar normal and tangential velocities back to vectors
        # and store into ball velocities
        self.ball1.velocity = v1n_new * normal + v1t * tangent
        self.ball2.velocity = v2n_new * normal + v2t * tangent

        # Check if any of the velocities have any -0's to get rid of
        _check_for_neg_zero()

    def _inelastic_collision_physics(self) -> None:
        """Calculate and update the physics of the balls after inelastic collision."""
        assert self.ball1
        assert self.ball2

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
        self.merged_ball = Ball(merged_params)

    def _process_post_collision_physics(self) -> None:
        """Calculate and update the physics of the balls after collision."""
        if self.collision_type == CollisionType.ELASTIC:
            self._elastic_collision_physics()
        # Else, inelastic collision
        else:
            self._inelastic_collision_physics()


    def _verify_conservation_of_momentum(self) -> None:
        """Verify that momentum is conserved after the collision."""
        assert self.total_momentum

        final_total_momentum: vp.vector

        # if the balls haven't merged, then check their momentums
        if not self.merged_ball:
            assert self.ball1
            assert self.ball2

            # Calculate final total momentum
            final_total_momentum = self.ball1.momentum + self.ball2.momentum
        # Else, check the merged ball momentum
        else:
            # Final total momentum is the momentum of the merged ball
            final_total_momentum = self.merged_ball.momentum

        # Verify momentum has been conserved
        assert round(self.total_momentum.mag, ndigits=3) == \
            round(final_total_momentum.mag, ndigits=3), \
            f'Initial total: {self.total_momentum.mag}, Final total: {final_total_momentum.mag}'

    def _verify_conservation_of_ke(self) -> None:
        """
        Verify that kinetic energy is conserved after the collision for elastic collisions,
        or calculate how much kinetic energy was lost after the collision for inelastic collisions.
        """
        assert self.init_total_ke is not None

        # if the balls haven't merged, then check their kinetic energy
        if not self.merged_ball:
            assert self.ball1
            assert self.ball2

            # Calculate final total ke
            self.final_total_ke = self.ball1.kinetic_energy + self.ball2.kinetic_energy

            # Calculate ke lost
            self.ke_lost = self.init_total_ke - self.final_total_ke

            # Verify KE has been conserved
            assert round(self.ke_lost, ndigits=3) == 0.0, \
                f'Initial total: {self.init_total_ke}, Final total: {self.final_total_ke}'
        # Else, check the loss of KE from the merged ball
        else:
            self.final_total_ke = self.merged_ball.kinetic_energy
            self.ke_lost = self.init_total_ke - self.final_total_ke
            print(f'Kinetic Energy lost in collision: {self.ke_lost:.2f} J')


    def _run_simulation(self) -> None:
        """Run the simulation loop."""
        assert self.ball1
        assert self.ball2

        dt: float = 0.01
        time_elapsed: float = 0.0

        SIMULATION_TIME_AFTER_COLLISION: int = 3  # secs

        while True:
            vp.rate(100)

            # If a collision hasn't occured already and the ball's positions are within the
            # distance of both radiuses, we have a collision
            if not self.collision_info and \
                   vp.mag(self.ball1.position - self.ball2.position) <= \
                   (self.ball1.radius + self.ball2.radius):
                # update balls based on physics of collision
                self._process_post_collision_physics()

                # Store collision state info for later
                self.collision_info = CollisionInfo(ball1=copy(self.ball1),
                                                    ball2=copy(self.ball2),
                                                    merged_ball=copy(self.merged_ball),
                                                    time=time_elapsed)

            # If we've had a collision, check if we've reached the "run a bit after the collision"
            # time. If we haven't had a collision, check if simulation duration has past
            if ((self.collision_info is not None) and
                time_elapsed > (self.collision_info.time + SIMULATION_TIME_AFTER_COLLISION)) or \
                    (time_elapsed > self.simulation_time):
                break

            if self.merged_ball:  # If we have a merged ball after collision
                self.merged_ball.update_position(dt)
            else:
                self.ball1.update_position(dt)
                self.ball2.update_position(dt)

            time_elapsed += dt

    def run(self) -> None:
        """
        Run the simulation.
        """
        self._init_simulation()

        assert self.ball1
        assert self.ball2
        assert self.ball1_state_t0
        assert self.ball2_state_t0
        assert self.dot_product is not None
        assert self.total_momentum
        assert self.init_total_ke is not None

        print('\n***************************************************')
        print('Initial Conditions:')
        self._print_ball_state([self.ball1_state_t0, self.ball2_state_t0])
        print(f'Initial Distance from each other: {self.initial_distance:.2f} m')
        print(f'Sum of Radii: {(self.ball1.radius + self.ball2.radius):.2f}')
        if self.dot_product < 0.0:
            print(f'Relative Speed toward each other: {self.relative_speed:.2f} m/s')
        else:
            print(f'Relative Speed away from each other: {self.relative_speed:.2f} m/s')
        print(f'Total Momentum: ({self.total_momentum.x:.2f}, {
            self.total_momentum.y:.2f}), or {vp.mag(self.total_momentum):.2f} N⋅s at {
            vp.degrees(vp.atan2(self.total_momentum.y, self.total_momentum.x)):.2f}°')
        print(f'Total Kinetic Energy: {(self.init_total_ke):.2f} J')
        print()

        # Run the simulation
        self._run_simulation()

        # If collision occured
        if self.collision_info is not None:
            print(f'Collision occured at time: {self.collision_info.time:.2f} secs')
            print('\nPost Collision Conditions:')
            if self.collision_info.merged_ball:
                self._print_ball_state([self.collision_info.merged_ball])
            else:
                self._print_ball_state([self.collision_info.ball1, self.collision_info.ball2])
        # Else no collision, see if the paths intersected
        else:
            print(f'No collision occured during simulation time of {
                self.simulation_time} secs.')

            # Calculate path intersection (if any)
            self._calculate_intersection()
            if self.intersection_info:
                print('Paths did intersect though:')
                print(f'  Path Intersection Point: ({self.intersection_info.position.x:.2f}, {
                    self.intersection_info.position.y:.2f})')
                print(f'  Time for Ball 1 to reach intersection: {
                    self.intersection_info.ball1_time:.2f} secs')
                print(f'  Time for Ball 2 to reach intersection: {
                    self.intersection_info.ball2_time:.2f} secs')
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
            - Simulation type (elastic or inelastic)  
            - Mass (kg) for Ball 1 and Ball 2  
            - Initial position (x, y) in meters for Ball 1 and Ball 2  
            - Initial velocity (vx, vy) in m/s for Ball 1 and Ball 2  
            - Simulation time in seconds  

    * What it does:  
        - Prints simulation details to the console.  
        - Runs the simulation, optionally displaying it in a VPython GUI window.  
        - Waits for a keypress to exit if the GUI is enabled.  

    """
    def _get_user_input() -> Tuple[PhysicsParameters, PhysicsParameters, float, CollisionType]:
        """
        Get user input for ball parameters, simulation time, and collision type.

        Returns:
            Tuple[PhysicsParameters, PhysicsParameters, float, CollisionType]: Parameters for both balls,
            simulation time, and collision type (elastic or inelastic).
        """
        def get_float(prompt: str) -> float:
            while True:
                try:
                    return float(input(prompt))
                except ValueError:
                    print("Please enter a valid number.")

        def get_vector(prompt: str) -> Tuple[float, float]:
            while True:
                try:
                    x, y = map(float, input(prompt).split(','))
                    return (x, y)
                except ValueError:
                    print("Please enter two numbers separated by a comma.")

        def get_collision_type() -> CollisionType:
            collision_selection = None
            while collision_selection not in ['e', 'i']:
                collision_selection = input(
                    "Enter collision type ('e'=elastic/'i'=inelastic): ").lower()

            if collision_selection == 'i':
                return CollisionType.INELASTIC
            else:
                return CollisionType.ELASTIC

        collision_type = get_collision_type()

        print("Enter parameters for Ball 1:")
        mass1: float = get_float("Mass (kg): ")
        position1: vp.vector = get_vector("Position (x,y) in meters: ")
        velocity1: vp.vector = get_vector("Velocity (x,y) in m/s: ")

        print("\nEnter parameters for Ball 2:")
        mass2: float = get_float("Mass (kg): ")
        position2: vp.vector = get_vector("Position (x,y) in meters: ")
        velocity2: vp.vector = get_vector("Velocity (x,y) in m/s: ")

        simulation_time: float = get_float("\nEnter simulation time (seconds): ")

        return (PhysicsParameters(mass1, position1, velocity1),
                PhysicsParameters(mass2, position2, velocity2),
                simulation_time,
                collision_type)

    parser = argparse.ArgumentParser(description='Ball Collision Simulator')
    parser.add_argument('--test', action='store_true', help='Run with pre-defined test case')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    if args.no_gui is True:
        BallCollisionSimulator.disable_gui(True)

    if args.test:
        # Pre-defined test case
        #                               Mass,  ( Position )   ( Velocity )
        ball1_params = PhysicsParameters(1.0,  (0.495, 0.0),  (0.0, 0.0))
        ball2_params = PhysicsParameters(1.0, (-0.495, -10.0), (0.0, 5.0))
        simulation_time: float = 10.0  # secs
        #collision_type = CollisionType.ELASTIC
        collision_type = CollisionType.INELASTIC
    else:
        # Get user input
        ball1_params, ball2_params, simulation_time, collision_type = _get_user_input()

    ball_collision_sim: BallCollisionSimulator = BallCollisionSimulator.create_simulator(
        ball1_params,
        ball2_params,
        simulation_time,
        collision_type
    )

    ball_collision_sim.run()

    if args.no_gui is False:
        print("Press any key to exit...")
        readchar.readkey()
        BallCollisionSimulator.quit_simulation()


if __name__ == '__main__':
    main()
