Module BallCollisionSimulator
=============================
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

Classes
-------

`Ball(params: BallParameters)`
:   Class representing a ball in the simulation.
    
    Initialize the Ball object.
    
    Args:
        params (BallParameters): Parameters for the ball.

    ### Instance variables

    `angle: float`
    :   Calculate and return the angle of the ball's velocity vector.

    `momentum: vp.vector`
    :   Calculate and return the momentum vector of the ball.

    `momentum_mag: float`
    :   Calculate and return the magnitude of the ball's momentum.

    `speed: float`
    :   Calculate and return the speed of the ball.

    ### Methods

    `update_position(self, dt: float) ‑> None`
    :   Update the position of the ball based on its velocity and time step.
        
        Args:
            dt (float): Time step for the update.

`BallCollisionSimulator(ball1_params: BallParameters, ball2_params: BallParameters, simulation_time: float)`
:   Class to simulate the collision between two balls.
    
    Initialize the BallCollisionSimulator.
    
    Args:
        ball1_params (BallParameters): Parameters for the first ball.
        ball2_params (BallParameters): Parameters for the second ball.
        simulation_time (float): Total time to simulate.

    ### Static methods

    `create_simulator(phys1_params: PhysicsParameters, phys2_params: PhysicsParameters, simulation_time: float) ‑> BallCollisionSimulator.BallCollisionSimulator`
    :   Create a BallCollisionSimulator instance with given parameters.
        
        Args:
            phys1_params (PhysicsParameters): Physics parameters for the first ball.
            phys2_params (PhysicsParameters): Physics parameters for the second ball.
            simulation_time (float): Total time to simulate.
        
        Returns:
            BallCollisionSimulator: An instance of the simulator.

    `quit_simulation() ‑> None`
    :   Stop the VPython server.

    ### Methods

    `remove_scene(self) ‑> None`
    :   Remove the VPython scene if it exists.

    `run(self) ‑> None`
    :   Run the entire simulation process.

`BallParameters(physics: PhysicsParameters, color: vp.color = vp.color.red, name: str = '')`
:   Class to store all parameters of a ball, including physics and visual properties.
    
    Initialize the BallParameters object.
    
    Args:
        physics (PhysicsParameters): Physical parameters of the ball.
        color (vp.color, optional): Color of the ball. Defaults to red.
        name (str, optional): Name or identifier for the ball. Defaults to ''.

`CollisionInfo(time: float, ball1: Ball, ball2: Ball)`
:   Data class to store information about a collision.

    ### Class variables

    `ball1: BallCollisionSimulator.Ball`
    :

    `ball2: BallCollisionSimulator.Ball`
    :

    `time: float`
    :

`IntersectionInfo(position: vp.vector, ball1_time: float, ball2_time: float)`
:   Data class to store information about an intersection of ball paths.

    ### Class variables

    `ball1_time: float`
    :

    `ball2_time: float`
    :

    `position: vp.vector`
    :

`PhysicsParameters(mass: float, position: Tuple[float, float], velocity: Tuple[float, float])`
:   Class to store the physical parameters of a ball.
    
    Initialize the PhysicsParameters object.
    
    Args:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Initial position of the ball (x, y) in meters.
        velocity (Tuple[float, float]): Initial velocity of the ball (vx, vy) in m/s.