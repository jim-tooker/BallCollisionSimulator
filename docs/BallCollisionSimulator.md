Module BallCollisionSimulator
=============================

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

    `momentum: vpython.vector`
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

    `create_simulator(phys1_params: BallCollisionSimulator.PhysicsParameters, phys2_params: BallCollisionSimulator.PhysicsParameters, simulation_time: float) ‑> BallCollisionSimulator.BallCollisionSimulator`
    :   Create a BallCollisionSimulator instance with given parameters.
        
        Args:
            phys1_params (PhysicsParameters): Physics parameters for the first ball.
            phys2_params (PhysicsParameters): Physics parameters for the second ball.
            simulation_time (float): Total time to simulate.
        
        Returns:
            BallCollisionSimulator: An instance of the simulator.

    ### Methods

    `quit_simulation(self) ‑> None`
    :   Stop the VPython server.

    `remove_scene(self) ‑> None`
    :   Remove the VPython scene if it exists.

    `run(self) ‑> None`
    :   Run the entire simulation process.

`BallParameters(physics: PhysicsParameters, color: vpython.color = vp.color.red name: str = '')`
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

`IntersectionInfo(position: vpython.vector, ball1_time: float, ball2_time: float)`
:   Data class to store information about an intersection of ball paths.

    ### Class variables

    `ball1_time: float`
    :

    `ball2_time: float`
    :

    `position: vpython.vector`
    :

`PhysicsParameters(mass: float, position: Tuple[float, float], velocity: Tuple[float, float])`
:   Class to store the physical parameters of a ball.
    
    Initialize the PhysicsParameters object.
    
    Args:
        mass (float): Mass of the ball in kg.
        position (Tuple[float, float]): Initial position of the ball (x, y) in meters.
        velocity (Tuple[float, float]): Initial velocity of the ball (vx, vy) in m/s.