"""
Defines the test cases used for the test_ball_collision() test.
"""
from typing import Tuple, Dict, List, Optional, Final
from dataclasses import dataclass
import math
from ball_collision_sim import PhysicsParameters, CollisionType, BallTrajectories

__author__ = "Jim Tooker"


DEGREES_TO_RADIANS = math.pi/180


@dataclass
class ExpectedBallState:
    """
    Dataclass to hold the expected state of a ball.
    
    Attributes:
        radius (Optional[float]): The expected radius of the ball.
        angle (Optional[float]): The expected angle of the ball's velocity vector.
        velocity (Optional[Tuple[float, float]]): The expected velocity of the ball.
        speed (Optional[float]): The expected speed of the ball.
        momentum (Optional[Tuple[float, float]]): The expected momentum of the ball.
        kinetic_energy (Optional[float]): The expected kinetic energy of the ball.
    """
    radius: Optional[float] = None
    angle: Optional[float] = None
    velocity: Optional[Tuple[float, float]] = None
    speed: Optional[float] = None
    momentum: Optional[Tuple[float, float]] = None
    kinetic_energy: Optional[float] = None


@dataclass
class ExpectedSimState:
    """
    Dataclass to hold the expected state of the simulation.
    
    Attributes:
        momentum (Optional[Tuple[float, float]]): The expected total momentum of the system.
        kinetic_energy (Optional[float]): The expected total kinetic energy of the system.
        relative_speed (Optional[float]): The expected relative speed between the balls.
        distance (Optional[float]): The expected distance between the balls.
        trajectories (Optional[BallTrajectories]): The expected trajectories of the balls.
        ke_lost (Optional[float]): The expected kinetic energy lost in an inelastic collision.
    """
    momentum: Optional[Tuple[float, float]] = None
    kinetic_energy: Optional[float] = None
    relative_speed: Optional[float] = None
    distance: Optional[float] = None
    trajectories: Optional[BallTrajectories] = None
    ke_lost: Optional[float] = None


@dataclass
class ExpectedCollisionInfo:
    """
    Dataclass to hold the expected collision information.
    
    Attributes:
        time (float): The expected time of collision.
        ball1_position (Optional[Tuple[float, float]]): The expected position of ball 1 at collision.
        ball2_position (Optional[Tuple[float, float]]): The expected position of ball 2 at collision.
        merged_position (Optional[Tuple[float, float]]): The expected position of the merged ball after
                                                         an inelastic collision.
    """
    time: float
    ball1_position: Optional[Tuple[float, float]] = None
    ball2_position: Optional[Tuple[float, float]] = None
    merged_position: Optional[Tuple[float, float]] = None


@dataclass
class ExpectedIntersectionInfo:
    """
    Dataclass to hold the expected intersection information.
    
    Attributes:
        position (Tuple[float, float]): The expected position of intersection.
        ball1_time (float): The expected time for ball 1 to reach the intersection.
        ball2_time (float): The expected time for ball 2 to reach the intersection.
    """
    position: Tuple[float, float]
    ball1_time: float
    ball2_time: float


@dataclass
class ExpectedResults:
    """
    Dataclass to hold all expected results for a test case.
    
    Attributes:
        init_sim (ExpectedSimState): Expected initial state of the simulation.
        final_balls (Optional[List[ExpectedBallState]]): Expected final states of the balls.
        merged_ball (Optional[ExpectedBallState]): Expected state of the merged ball in an inelastic collision.
        final_sim (ExpectedSimState): Expected final state of the simulation.
        collision_info (Optional[ExpectedCollisionInfo]): Expected collision information.
        intersect_info (Optional[ExpectedIntersectionInfo]): Expected intersection information.
    """
    init_sim: ExpectedSimState
    final_sim: ExpectedSimState
    final_balls: Optional[List[ExpectedBallState]] = None
    merged_ball: Optional[ExpectedBallState] = None
    collision_info: Optional[ExpectedCollisionInfo] = None
    intersect_info: Optional[ExpectedIntersectionInfo] = None


TESTS: Final[Dict[str, Tuple[List[PhysicsParameters], float, CollisionType, Optional[float], ExpectedResults]]] = {
    "test01_elastic_collision_head_on_x_axis": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-1, 0)),
                         ExpectedBallState(velocity=(1, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1,
                                                 ball1_position=(-0.5, 0),
                                                 ball2_position=(0.5, 0)),
            intersect_info=None
        )
    ),
    "test02_elastic_collision_converging_diagonal": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 1)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-1, 1)),
                         ExpectedBallState(velocity=(1, 1))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1,
                                                 ball1_position=(-0.5, 1),
                                                 ball2_position=(0.5, 1)),
            intersect_info=None
        )
    ),
    "test03_elastic_miss_diverging_diagonal": (
        [PhysicsParameters(mass=1, position=(-1, 0), velocity=(-1, 1)),
         PhysicsParameters(mass=1, position=(1, 0), velocity=(1, 1))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=None
        )
    ),
    "test04_elastic_collision_diagonal_opposing_symmetry": (
        [PhysicsParameters(mass=1, position=(-1.5, -1.5), velocity=(1, 1)),
         PhysicsParameters(mass=1, position=(1.5, 1.5), velocity=(-1, -1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-1, -1)),
                         ExpectedBallState(velocity=(1, 1))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1.15,
                                                 ball1_position=(-0.35, -0.35),
                                                 ball2_position=(0.35, 0.35)),
            intersect_info=None
        )
    ),
    "test05_elastic_intersection_right_triangle": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 1)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(0, 2))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(1.5, 3),
                                                    ball1_time=3,
                                                    ball2_time=1.5)
        )
    ),
    "test06_elastic_collision_right_triangle": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 1)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(0, 1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(0, 1)),
                         ExpectedBallState(velocity=(1, 1))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=2,
                                                 ball1_position=(0.5, 2),
                                                 ball2_position=(1.5, 2)),
            intersect_info=None
        )
    ),
    "test07_elastic_intersection_small_angle_trajectories": (
        [PhysicsParameters(mass=0.01, position=(-1, 0), velocity=(math.cos(45*DEGREES_TO_RADIANS),
                                                                  math.sin(45*DEGREES_TO_RADIANS))),
         PhysicsParameters(mass=0.01, position=(0, 0), velocity=(math.cos(46*DEGREES_TO_RADIANS),
                                                                 math.sin(46*DEGREES_TO_RADIANS)))],
        45,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(28.145, 29.145),
                                                    ball1_time=41.217,
                                                    ball2_time=40.516)
        )
    ),
    "test08_elastic_miss_parallel_same_direction": (
        [PhysicsParameters(mass=1, position=(0, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(0, 2), velocity=(1, 0))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            collision_info=None,
            intersect_info=None
        )
    ),
    "test09_elastic_miss_parallel_opposing": (
        [PhysicsParameters(mass=1, position=(0, -2), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(0, 2), velocity=(-1, 0))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=None
        )
    ),
    "test10_elastic_miss_parallel_velocity_differential": (
        [PhysicsParameters(mass=1, position=(0, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(0, 2), velocity=(2, 0))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=None
        )
    ),
    "test11_elastic_collision_overtaking": (
        [PhysicsParameters(mass=1, position=(0, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(-2.5, 0), velocity=(2, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(2, 0)),
                         ExpectedBallState(velocity=(1, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1.5,
                                                 ball1_position=(1.5, 0),
                                                 ball2_position=(0.5, 0)),
            intersect_info=None
        )
    ),
    "test12_elastic_collision_different_masses": (
        [PhysicsParameters(mass=1, position=(-5, 0), velocity=(2, 0)),
         PhysicsParameters(mass=4, position=(5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-2.8, 0)),
                         ExpectedBallState(velocity=(0.2, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=2.91,
                                                 ball1_position=(0.82, 0),
                                                 ball2_position=(2.09, 0)),
            intersect_info=None
        )
    ),
    "test13_elastic_collision_stationary_target": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(0, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(0, 0)),
                         ExpectedBallState(velocity=(1, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=2,
                                                 ball1_position=(0.5, 0),
                                                 ball2_position=(1.5, 0)),
            intersect_info=None
        )
    ),
    "test14_elastic_miss_dual_stationary_objects": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(0, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(0, 0))],
        5,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            collision_info=None,
            intersect_info=None
        )
    ),
    "test15_elastic_collision_same_position_opposing_velocities": (
        [PhysicsParameters(mass=1, position=(1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=[ExpectedBallState(velocity=(-1, 0)),
                         ExpectedBallState(velocity=(1, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=0,
                                                 ball1_position=(1.5, 0),
                                                 ball2_position=(1.5, 0)),
            intersect_info=None
        )
    ),
    "test16_elastic_collision_glancing_blow": (
        [PhysicsParameters(mass=1, position=(3, 4), velocity=(-1, -0.5)),
         PhysicsParameters(mass=1, position=(-3.3, -4), velocity=(0, 1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(0.05172, -0.4628)),
                         ExpectedBallState(velocity=(-1.0517, 0.96282))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=5.31,
                                                 ball1_position=(-2.31, 1.345),
                                                 ball2_position=(-3.3, 1.31)),
            intersect_info=None
        )
    ),
    "test17_elastic_collision_oblique_angle": (
        [PhysicsParameters(mass=1, position=(0.4, 3), velocity=(0, -2)),
         PhysicsParameters(mass=2, position=(-0.5, -3), velocity=(0, 1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(1.9075, -0.60116)),
                         ExpectedBallState(velocity=(-0.95376, 0.30058))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1.78,
                                                 ball1_position=(0.40, -0.56),
                                                 ball2_position=(-0.50, -1.22)),
            intersect_info=None
        )
    ),
    "test18_elastic_intersection_diagonal_trajectories": (
        [PhysicsParameters(mass=1, position=(3, 4), velocity=(-1, -1)),
         PhysicsParameters(mass=1, position=(-3.3, -4), velocity=(0, 1))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(-3.3, -2.3),
                                                    ball1_time=6.3,
                                                    ball2_time=1.7)
        )
    ),
    "test19_elastic_collision_side_pocket_pool_shot": (
        [PhysicsParameters(mass=1, position=(0.495, 0), velocity=(0, 0)),
         PhysicsParameters(mass=1, position=(-0.495, -10), velocity=(0, 5))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(0.5, 0.0505)),
                         ExpectedBallState(velocity=(-0.5, 4.9495))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            collision_info=ExpectedCollisionInfo(time=1.98,
                                                 ball1_position=(0.495, 0),
                                                 ball2_position=(-0.495, -0.1)),
            intersect_info=None
        )
    ),
    "test20_inelastic_collision_head_on_symmetry": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(0, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=1),
            collision_info=ExpectedCollisionInfo(time=1,
                                                 merged_position=(0, 0)),
            intersect_info=None
        )
    ),
    "test21_inelastic_collision_different_masses": (
        [PhysicsParameters(mass=1, position=(-3, 0), velocity=(1, 0)),
         PhysicsParameters(mass=2, position=(3, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(-0.333, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=1.333),
            collision_info=ExpectedCollisionInfo(time=2.44,
                                                 merged_position=(0.187, 0)),
            intersect_info=None
        )
    ),
    "test22_inelastic_collision_glancing_blow": (
        [PhysicsParameters(mass=1, position=(0.4, 3), velocity=(0, -2)),
         PhysicsParameters(mass=1, position=(-0.5, -3), velocity=(0, 1))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(0, -0.5)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=2.25),
            collision_info=ExpectedCollisionInfo(time=1.86,
                                                 merged_position=(-0.05, -0.93)),
            intersect_info=None
        )
    ),
    "test23_inelastic_collision_stationary_target": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(0, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(0.5, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=0.25),
            collision_info=ExpectedCollisionInfo(time=2,
                                                 merged_position=(1, 0)),
            intersect_info=None
        )
    ),
    "test24_inelastic_collision_side_pocket_pool_shot": (
        [PhysicsParameters(mass=1, position=(0.495, 0), velocity=(0, 0)),
         PhysicsParameters(mass=1, position=(-0.495, -10), velocity=(0, 5))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(0, 2.5)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=6.25),
            collision_info=ExpectedCollisionInfo(time=1.98,
                                                 merged_position=(0, -0.05)),
            intersect_info=None
        )
    ),
    "test25_inelastic_collision_diagonal_approach": (
        [PhysicsParameters(mass=1, position=(-5, -5), velocity=(1, 1)),
         PhysicsParameters(mass=2, position=(5, 5), velocity=(-2, -2))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(-1, -1)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=6),
            collision_info=ExpectedCollisionInfo(time=3.07,
                                                 merged_position=(-1.403, -1.403)),
            intersect_info=None
        )
    ),
    "test26_inelastic_collision_same_position_different_velocities": (
        [PhysicsParameters(mass=1, position=(2, 0), velocity=(2, 0)),
         PhysicsParameters(mass=1, position=(2, 0), velocity=(-4, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(-1, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=9),
            collision_info=ExpectedCollisionInfo(time=0,
                                                 merged_position=(2, 0)),
            intersect_info=None
        )
    ),
    "test27_elastic_collision_same_position_same_velocity": (
        [PhysicsParameters(mass=1, position=(1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(1, 0))],
        10,  # simulation time (s)
        CollisionType.ELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            final_balls=[ExpectedBallState(velocity=(1, 0)),
                         ExpectedBallState(velocity=(1, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.CONSTANT),
            collision_info=ExpectedCollisionInfo(time=0,
                                                 ball1_position=(1.5, 0),
                                                 ball2_position=(1.5, 0)),
            intersect_info=None
        )
    ),
    "test28_inelastic_collision_different_velocities": (
        [PhysicsParameters(mass=1, position=(-5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(10, 0), velocity=(-10, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(-4.5, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=30.25),
            collision_info=ExpectedCollisionInfo(time=1.28,
                                                 merged_position=(-3.26, 0)),
            intersect_info=None
        )
    ),
    "test29_inelastic_collision_different_masses_and_velocities": (
        [PhysicsParameters(mass=1, position=(-10, 0), velocity=(3, 0)),
         PhysicsParameters(mass=10, position=(10, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(-0.63636, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=7.2727),
            collision_info=ExpectedCollisionInfo(time=4.61,
                                                 merged_position=(5.2482, 0)),
            intersect_info=None
        )
    ),
    "test30_inelastic_collision_small_fast_ball_wins": (
        [PhysicsParameters(mass=1, position=(-10, 0), velocity=(10, 0)),
         PhysicsParameters(mass=5, position=(8, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.INELASTIC,
        None,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=None,
            merged_ball=ExpectedBallState(velocity=(0.8333, 0)),
            final_sim=ExpectedSimState(trajectories=BallTrajectories.MERGED,
                                       ke_lost=50.417),
            collision_info=ExpectedCollisionInfo(time=1.52,
                                                 merged_position=(6.267, 0)),
            intersect_info=None
        )
    ),
    "test31_partial_collision_head_on_x_axis": (
        [PhysicsParameters(mass=1, position=(-1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.PARTIAL,
        0.5, # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-0.5, 0)),
                         ExpectedBallState(velocity=(0.5, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0.75),
            collision_info=ExpectedCollisionInfo(time=1,
                                                 ball1_position=(-0.5, 0),
                                                 ball2_position=(0.5, 0)),
            intersect_info=None
        )
    ),
    "test32_partial_collision_side_pocket_pool_shot": (
        [PhysicsParameters(mass=1, position=(0.495, 0), velocity=(0, 0)),
         PhysicsParameters(mass=1, position=(-0.495, -10), velocity=(0, 5))],
        10,  # simulation time (s)
        CollisionType.PARTIAL,
        0.8,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(0.44995, 0.04545)),
                         ExpectedBallState(velocity=(-0.44995, 4.9546))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0.0227),
            collision_info=ExpectedCollisionInfo(time=1.98,
                                                 ball1_position=(0.495, 0),
                                                 ball2_position=(-0.495, -0.1)),
            intersect_info=None
        )
    ),
    "test33_partial_collision_different_masses_and_velocities": (
        [PhysicsParameters(mass=1, position=(-5, 0), velocity=(2, 0)),
         PhysicsParameters(mass=4, position=(5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.PARTIAL,
        0.2,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-0.88, 0)),
                         ExpectedBallState(velocity=(-0.28, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=3.456),
            collision_info=ExpectedCollisionInfo(time=2.91,
                                                 ball1_position=(0.82, 0),
                                                 ball2_position=(2.09, 0)),
            intersect_info=None
        )
    ),
    "test34_partial_collision_same_position_opposing_velocities": (
        [PhysicsParameters(mass=1, position=(1.5, 0), velocity=(1, 0)),
         PhysicsParameters(mass=1, position=(1.5, 0), velocity=(-1, 0))],
        10,  # simulation time (s)
        CollisionType.PARTIAL,
        0.5,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING),
            final_balls=[ExpectedBallState(velocity=(-0.5, 0)),
                         ExpectedBallState(velocity=(0.5, 0))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0.75),
            collision_info=ExpectedCollisionInfo(time=0,
                                                 ball1_position=(1.5, 0),
                                                 ball2_position=(1.5, 0)),
            intersect_info=None
        )
    ),
    "test35_partial_collision_diagonal_opposing_symmetry": (
        [PhysicsParameters(mass=1, position=(-1.5, -1.5), velocity=(1, 1)),
         PhysicsParameters(mass=1, position=(1.5, 1.5), velocity=(-1, -1))],
        10,  # simulation time (s)
        CollisionType.PARTIAL,
        0.7,  # COR
        ExpectedResults(
            init_sim=ExpectedSimState(trajectories=BallTrajectories.CONVERGING),
            final_balls=[ExpectedBallState(velocity=(-0.7, -0.7)),
                         ExpectedBallState(velocity=(0.7, 0.7))],
            merged_ball=None,
            final_sim=ExpectedSimState(trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=1.02),
            collision_info=ExpectedCollisionInfo(time=1.15,
                                                 ball1_position=(-0.35, -0.35),
                                                 ball2_position=(0.35, 0.35)),
            intersect_info=None
        )
    ),
}
"""
TESTS: Contains all the test cases definitions
"""


ACTIVE_TESTS: Final[List[str]] = [
    "all",   # This will run everything
    # "test01_elastic_collision_head_on_x_axis",
    # "test02_elastic_collision_converging_diagonal",
    # "test03_elastic_miss_diverging_diagonal",
    # "test04_elastic_collision_diagonal_opposing_symmetry",
    # "test05_elastic_intersection_right_triangle",
    # "test06_elastic_collision_right_triangle",
    # "test07_elastic_intersection_small_angle_trajectories",
    # "test08_elastic_miss_parallel_same_direction",
    # "test09_elastic_miss_parallel_opposing",
    # "test10_elastic_miss_parallel_velocity_differential",
    # "test11_elastic_collision_overtaking",
    # "test12_elastic_collision_different_masses",
    # "test13_elastic_collision_stationary_target",
    # "test14_elastic_miss_dual_stationary_objects",
    # "test15_elastic_collision_same_position_opposing_velocities",
    # "test16_elastic_collision_glancing_blow",
    # "test17_elastic_collision_oblique_angle",
    # "test18_elastic_intersection_diagonal_trajectories",
    # "test19_elastic_collision_side_pocket_pool_shot",
    # "test20_inelastic_collision_head_on_symmetry",
    # "test21_inelastic_collision_different_masses",
    # "test22_inelastic_collision_glancing_blow",
    # "test23_inelastic_collision_stationary_target",
    # "test24_inelastic_collision_side_pocket_pool_shot",
    # "test25_inelastic_collision_diagonal_approach",
    # "test26_inelastic_collision_same_position_different_velocities",
    # "test27_elastic_collision_same_position_same_velocity",
    # "test28_inelastic_collision_different_velocities",
    # "test29_inelastic_collision_different_masses_and_velocities",
    # "test30_inelastic_collision_small_fast_ball_wins",
    # "test31_partial_collision_head_on_x_axis",
    # "test32_partial_collision_side_pocket_pool_shot",
    # "test33_partial_collision_different_masses_and_velocities",
    # "test34_partial_collision_same_position_opposing_velocities",
    # "test35_partial_collision_diagonal_opposing_symmetry",
]
"""
ACTIVE_TESTS: Defines the active tests (what tests will be run)
"""
