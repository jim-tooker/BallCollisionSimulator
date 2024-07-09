"""
Ball Collision Simulator

This module simulates the elastic collision between two balls using the VPython library. It
contains classes to define the physical and visual properties of the balls, perform the 
simulation, and visualize the collision in a 3D space.

Classes:
    - PhysicsParameters: Stores the physical parameters of a ball.
    - BallParameters: Combines physical and visual properties of a ball.
    - Ball: Represents a ball in the simulation, managing its state and visualization.
    - CollisionInfo: Data class for storing collision information.
    - IntersectionInfo: Data class for storing intersection information of ball paths.
    - BallCollisionSimulator: Manages the entire simulation process, including initialization, 
      running the simulation, and handling collisions.

Functions:
    - get_user_input: Prompts the user to input parameters for the balls and the simulation time.
    - main: The main entry point of the program, running the simulation with either predefined 
      test parameters or user input.

Usage:
    Run this module as a script to start the simulation. The user can choose to input custom
    parameters for the balls and the simulation time or use predefined test parameters with 
    the --test argument.
"""
from __future__ import annotations
from typing import Tuple, Optional
import argparse
from dataclasses import dataclass
from copy import copy
import vpython as vp
import readchar

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

class BallParameters:
    """
    Class to store all parameters of a ball, including physics and visual properties.
    """
    def __init__(self, physics: PhysicsParameters, color: vp.color = vp.color.red, name: str = ''):
        """
        Initialize the BallParameters object.

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
    """
    def __init__(self, params: BallParameters):
        """
        Initialize the Ball object.

        Args:
            params (BallParameters): Parameters for the ball.
        """
        self.mass: float = params.mass
        self.position: vp.vector = params.position
        self.velocity: vp.vector = params.velocity

        # Radius proportional to the mass (0.5m for 1 kg)
        self.radius: float = 0.5 * (self.mass ** (1/3))

        self.name: str = params.name
        self.sphere: vp.sphere = vp.sphere(pos=self.position,
                                           radius=self.radius,
                                           color=params.color,
                                           make_trail=True)
        self.label: vp.label = vp.label(pos=self.position,
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

    def update_position(self, dt: float) -> None:
        """
        Update the position of the ball based on its velocity and time step.

        Args:
            dt (float): Time step for the update.
        """
        self.position += self.velocity * dt
        self.sphere.pos = self.position
        self.label.pos = self.position

@dataclass
class CollisionInfo:
    """Data class to store information about a collision."""
    time: float
    ball1: Ball
    ball2: Ball

@dataclass
class IntersectionInfo:
    """Data class to store information about an intersection of ball paths."""
    position: vp.vector
    ball1_time: float
    ball2_time: float

class BallCollisionSimulator:
    """
    Class to simulate the collision between two balls.
    """
    def __init__(self, ball1_params: BallParameters,
                 ball2_params: BallParameters,
                 simulation_time: float):
        """
        Initialize the BallCollisionSimulator.

        Args:
            ball1_params (BallParameters): Parameters for the first ball.
            ball2_params (BallParameters): Parameters for the second ball.
            simulation_time (float): Total time to simulate.
        """
        self.ball1_params: BallParameters = ball1_params
        self.ball2_params: BallParameters = ball2_params
        self.simulation_time: float = simulation_time
        self.scene: Optional[vp.canvas] = None
        self.ball1: Optional[Ball] = None
        self.ball2: Optional[Ball] = None
        self.ball1_state_t0: Optional[Ball] = None
        self.ball2_state_t0: Optional[Ball] = None
        self.initial_distance: Optional[float] = None
        self.total_momentum: Optional[vp.vector] = None
        self.dot_product: Optional[float] = None
        self.relative_speed: Optional[float] = None
        self.collision_info: Optional[CollisionInfo] = None
        self.intersection_info: Optional[IntersectionInfo] = None

    @staticmethod
    def create_simulator(phys1_params: PhysicsParameters,
                         phys2_params: PhysicsParameters,
                         simulation_time: float) -> BallCollisionSimulator:
        """
        Create a BallCollisionSimulator instance with given parameters.

        Args:
            phys1_params (PhysicsParameters): Physics parameters for the first ball.
            phys2_params (PhysicsParameters): Physics parameters for the second ball.
            simulation_time (float): Total time to simulate.

        Returns:
            BallCollisionSimulator: An instance of the simulator.
        """
        ball1_params : BallParameters = BallParameters(phys1_params, color=vp.color.blue, name='1')
        ball2_params : BallParameters = BallParameters(phys2_params, color=vp.color.red, name='2')

        return BallCollisionSimulator(ball1_params, ball2_params, simulation_time)

    def remove_scene(self) -> None:
        """Remove the VPython scene if it exists."""
        if self.scene:
            self.scene.delete()
            self.scene = None

    @staticmethod
    def quit_simulation() -> None:
        """Stop the VPython server."""
        # We don't import vp_services until needed, because importing it will start
        # the server, if not started already.
        import vpython.no_notebook as vp_services

        vp_services.stop_server()

    def __init_simulation(self) -> None:
        """Initialize the simulation by setting up the scene and balls."""
        self.scene = vp.canvas(title='Elastic Collision Simulation',
                               width=800,
                               height=800,
                               center=vp.vector(0, 0, 0),
                               background=vp.color.black)

        # Set up grid
        self.__create_grid_and_axes()

        # Create ball objects
        self.ball1 = Ball(self.ball1_params)
        self.ball2 = Ball(self.ball2_params)

        # Store intial state for later
        self.ball1_state_t0 = copy(self.ball1)
        self.ball2_state_t0 = copy(self.ball2)

        # Calculate initial momentum
        self.total_momentum = self.ball1.momentum + self.ball2.momentum

        # Calculate initial distance
        self.initial_distance = vp.mag(
            self.ball1.position - self.ball2.position)

        # Calculate the relative speed of the balls to each other
        self.dot_product = vp.dot(self.ball1.velocity - self.ball2.velocity, 
                                  self.ball1.position - self.ball2.position)
        self.relative_speed = (self.ball1.velocity - self.ball2.velocity).mag

    def __create_grid_and_axes(self) -> None:
        """Create a grid and axes for the simulation scene."""
        GRID_RANGE : int = 10
        step : int = 1

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
    def __print_velocity_details(ball1: Ball, ball2: Ball) -> None:
        """
        Print detailed information about the velocities of two balls.

        Args:
            ball1 (Ball): First ball object.
            ball2 (Ball): Second ball object.
        """
        print()
        for ball in ball1, ball2:
            print(f'Ball {ball.name}:')
            print(f'  Mass: {ball.mass} kg')
            print(f'  Radius: {ball.radius:.2f} m')
            print(f'  Position: ({ball.position.x:.2f}, {
                  ball.position.y:.2f})')
            print(f'  Velocity: ({ball.velocity.x:.2f}, {ball.velocity.y:.2f}), or {
                ball.speed:.2f} m/s at {ball.angle:.2f} degrees')
            print(f'  Momentum: ({ball.momentum.x:.2f}, {ball.momentum.y:.2f}), or {
                ball.momentum_mag:.2f} N-s at {ball.angle:.2f} degrees')
        print()

    def __calculate_intersection(self) -> None:
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

    def __process_post_collision_physics(self) -> None:
        """Calculate and update the velocities of the balls after collision."""
        assert self.ball1
        assert self.ball2

        m1 : float = self.ball1.mass
        m2 : float = self.ball2.mass
        v1_initial : vp.vector = self.ball1.velocity
        v2_initial : vp.vector = self.ball2.velocity
        self.ball1.velocity = (v1_initial * (m1 - m2) +
                            2 * m2 * v2_initial) / (m1 + m2)
        self.ball2.velocity = (v2_initial * (m2 - m1) +
                            2 * m1 * v1_initial) / (m1 + m2)

    def __verify_conservation_of_momentum(self) -> None:
        """Verify that momentum is conserved after the collision."""
        assert self.ball1
        assert self.ball2
        assert self.total_momentum

        # Calculate final total momentum
        final_total_momentum : vp.vector = self.ball1.momentum + self.ball2.momentum

        # Verify momentum has been conserved
        assert round(self.total_momentum.mag, ndigits=3) == \
               round(final_total_momentum.mag, ndigits=3), \
               f'Initial total: {self.total_momentum.mag}, Final total: {final_total_momentum.mag}'

    def __run_simulation(self) -> None:
        """Run the main simulation loop."""
        assert self.ball1
        assert self.ball2

        dt : float = 0.01
        time_elapsed : float = 0.0

        SIMULATION_TIME_AFTER_COLLISION : int = 2  # secs

        while True:
            vp.rate(100)

            # If a collision hasn't occured already and the ball's positions are within the
            # distance of both radiuses, we have a collision
            if not self.collision_info and \
                    vp.mag(self.ball1.position - self.ball2.position) <= \
                        (self.ball1.radius + self.ball2.radius):
                # update balls based on physics of collision
                self.__process_post_collision_physics()

                # Store collision state info for later
                self.collision_info = CollisionInfo(ball1=copy(self.ball1),
                                                    ball2=copy(self.ball2),
                                                    time=time_elapsed)

            # If we've had a collision, check if we've reached the "run a bit after the collision"
            # time. If we haven't had a collision, check if simulation duration has past
            if ((self.collision_info is not None) and
                time_elapsed > (self.collision_info.time + SIMULATION_TIME_AFTER_COLLISION)) or \
                    (time_elapsed > self.simulation_time):
                break

            self.ball1.update_position(dt)
            self.ball2.update_position(dt)
            time_elapsed += dt

    def run(self) -> None:
        """Run the entire simulation process."""
        self.__init_simulation()

        assert self.ball1
        assert self.ball2
        assert self.ball1_state_t0
        assert self.ball2_state_t0
        assert self.dot_product is not None
        assert self.total_momentum

        print('\n***************************************************')
        print('Initial Conditions:')
        self.__print_velocity_details(self.ball1_state_t0, self.ball2_state_t0)
        print(f'Initial Distance from each other: {self.initial_distance:.2f} m')
        print(f'Sum of radii for both balls: {(self.ball1.radius + self.ball2.radius):.2f}')
        if self.dot_product < 0.0:
            print(f'Relative Speed toward each other: {self.relative_speed:.2f} m/s')
        else:
            print(f'Relative Speed away from each other: {self.relative_speed:.2f} m/s')
        print(f'Total Momentum: ({self.total_momentum.x:.2f}, {
            self.total_momentum.y:.2f}), or {vp.mag(self.total_momentum):.2f} m/s at {
            vp.degrees(vp.atan2(self.total_momentum.y, self.total_momentum.x)):.2f} degrees')
        print()

        # Run the simulation
        self.__run_simulation()

        # If collision occured
        if self.collision_info is not None:
            print(f'Collision occured at time: {self.collision_info.time:.2f} secs')
            print('\nPost Collision Conditions:')
            self.__print_velocity_details(
                self.collision_info.ball1, self.collision_info.ball2)
        # Else no collision, see if the paths intersected
        else:
            print(f'No collision occured during simulation time of {
                self.simulation_time} secs.')

            # Calculate path intersection (if any)
            self.__calculate_intersection()
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

        print('###################################################')

        self.__verify_conservation_of_momentum()


if __name__ == '__main__':
    def get_user_input() -> Tuple[PhysicsParameters, PhysicsParameters, float]:
        """
        Get user input for ball parameters and simulation time.

        Returns:
            Tuple[PhysicsParameters, PhysicsParameters, float]: Parameters for both balls
            and simulation time.
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

        print("Enter parameters for Ball 1:")
        mass1 : float = get_float("Mass (kg): ")
        position1 : vp.vector = get_vector("Position (x,y) in meters: ")
        velocity1 : vp.vector = get_vector("Velocity (x,y) in m/s: ")

        print("\nEnter parameters for Ball 2:")
        mass2 : float = get_float("Mass (kg): ")
        position2 : vp.vector = get_vector("Position (x,y) in meters: ")
        velocity2 : vp.vector = get_vector("Velocity (x,y) in m/s: ")

        simulation_time : float = get_float("\nEnter simulation time (seconds): ")

        return (PhysicsParameters(mass1, position1, velocity1),
                PhysicsParameters(mass2, position2, velocity2),
                simulation_time)

    parser = argparse.ArgumentParser(description='Ball Collision Simulator')
    parser.add_argument('--test', action='store_true', help='Run with pre-defined test case')
    args = parser.parse_args()

    if args.test:
        # Pre-defined test case
        #                                Mass, ( Position )  ( Velocity )
        ball1_params = PhysicsParameters( 1.0, ( 3.0,  4.0), (-3.0, -0.1))
        ball2_params = PhysicsParameters( 4.0, (-3.3, -4.0), ( 0.0,  4.0))
        simulation_time: float = 10.0  # secs
    else:
        # Get user input
        ball1_params, ball2_params, simulation_time = get_user_input()

    ball_collision_sim: BallCollisionSimulator = BallCollisionSimulator.create_simulator(
        ball1_params,
        ball2_params,
        simulation_time
    )

    ball_collision_sim.run()

    print("Press any key to exit...")
    readchar.readkey()
    ball_collision_sim.remove_scene()
    BallCollisionSimulator.quit_simulation()
