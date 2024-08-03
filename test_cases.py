"""
Defines the test cases used for the test_ball_collision() test.
"""
from typing import Tuple, Dict, List, Iterable, Optional, Final
from dataclasses import dataclass, field
import math
from ball_collision_sim import PhysicsParameters, CollisionType, BallTrajectories

__author__ = "Jim Tooker"


DEGREES_TO_RADIANS = math.pi/180.0


ACTIVE_TESTS: Final[List[str]] = [

    # Notes on categories of tests:
    #     • Head-on collisions  (01, 20)
    #     • Diagonal collisions (02, 04, 25)
    #     • Parallel trajectories (08, 09, 10)
    #     • Glancing blows      (16, 17, 22)
    #     • Same-position starts (15, 26, 27)
    #     • Special cases       (11-overtaking, 19-pool shot)
    #     • Stationary object   (13, 23)
    #     • Intersections       (05, 07, 18)
    #     • Different masses    (12, 21)


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
]


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
    radius: Optional[float] = field(default=None)
    angle: Optional[float] = field(default=None)
    velocity: Optional[Tuple[float, float]] = field(default=None)
    speed: Optional[float] = field(default=None)
    momentum: Optional[Tuple[float, float]] = field(default=None)
    kinetic_energy: Optional[float] = field(default=None)


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
    momentum: Optional[Tuple[float, float]] = field(default=None)
    kinetic_energy: Optional[float] = field(default=None)
    relative_speed: Optional[float] = field(default=None)
    distance: Optional[float] = field(default=None)
    trajectories: Optional[BallTrajectories] = field(default=None)
    ke_lost: Optional[float] = field(default=None)


@dataclass
class ExpectedCollisionInfo:
    """
    Dataclass to hold the expected collision information.
    
    Attributes:
        time (Optional[float]): The expected time of collision.
        ball1_position (Optional[Tuple[float, float]]): The expected position of ball 1 at collision.
        ball2_position (Optional[Tuple[float, float]]): The expected position of ball 2 at collision.
        merged_position (Optional[Tuple[float, float]]): The expected position of the merged ball after
                                                         an inelastic collision.
    """
    time: Optional[float] = field(default=None)
    ball1_position: Optional[Tuple[float, float]] = field(default=None)
    ball2_position: Optional[Tuple[float, float]] = field(default=None)
    merged_position: Optional[Tuple[float, float]] = field(default=None)


@dataclass
class ExpectedIntersectionInfo:
    """
    Dataclass to hold the expected intersection information.
    
    Attributes:
        position (Optional[Tuple[float, float]]): The expected position of intersection.
        ball1_time (Optional[float]): The expected time for ball 1 to reach the intersection.
        ball2_time (Optional[float]): The expected time for ball 2 to reach the intersection.
    """
    position: Optional[Tuple[float, float]] = field(default=None)
    ball1_time: Optional[float] = field(default=None)
    ball2_time: Optional[float] = field(default=None)


@dataclass
class ExpectedResults:
    """
    Dataclass to hold all expected results for a test case.
    
    Attributes:
        init_balls (Iterable[ExpectedBallState]): Expected initial states of the balls.
        init_sim (ExpectedSimState): Expected initial state of the simulation.
        final_balls (Optional[Iterable[ExpectedBallState]]): Expected final states of the balls.
        merged_ball (Optional[ExpectedBallState]): Expected state of the merged ball in an inelastic collision.
        final_sim (ExpectedSimState): Expected final state of the simulation.
        collision_info (Optional[ExpectedCollisionInfo]): Expected collision information.
        intersect_info (Optional[ExpectedIntersectionInfo]): Expected intersection information.
    """
    init_balls: Iterable[ExpectedBallState] = field(default_factory=list)
    init_sim: ExpectedSimState = field(default_factory=ExpectedSimState)

    final_balls: Optional[Iterable[ExpectedBallState]] = field(default_factory=Iterable[ExpectedBallState])
    merged_ball: Optional[ExpectedBallState] = field(default_factory=ExpectedBallState)
    final_sim: ExpectedSimState = field(default_factory=ExpectedSimState)

    collision_info: Optional[ExpectedCollisionInfo] = None
    intersect_info: Optional[ExpectedIntersectionInfo] = None


TESTS: Final[Dict[str, Tuple[PhysicsParameters, PhysicsParameters, float, CollisionType, ExpectedResults]]] = {
    "test01_elastic_collision_head_on_x_axis": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=180,
                                          speed=1,
                                          momentum=(-1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=1,
                                      relative_speed=2,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=180,
                                           speed=1,
                                           velocity=(-1, 0),
                                           momentum=(-1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=1,
                                       relative_speed=2,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1,
                                                 ball1_position=(-0.5, 0),
                                                 ball2_position=(0.5, 0)),
            intersect_info=None
        )
    ),
    "test02_elastic_collision_converging_diagonal": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (-1.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=135,
                                          speed=1.414,
                                          momentum=(-1, 1),
                                          kinetic_energy=1)],
            init_sim=ExpectedSimState(momentum=(0, 2),
                                      kinetic_energy=2,
                                      relative_speed=2,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=135,
                                           speed=1.414,
                                           velocity=(-1, 1),
                                           momentum=(-1, 1),
                                           kinetic_energy=1),
                         ExpectedBallState(angle=45,
                                           speed=1.414,
                                           velocity=(1, 1),
                                           momentum=(1, 1),
                                           kinetic_energy=1)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 2),
                                       kinetic_energy=2,
                                       relative_speed=2,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1,
                                                 ball1_position=(-0.5, 1),
                                                 ball2_position=(0.5, 1)),
            intersect_info=None
        )
    ),
    "test03_elastic_miss_diverging_diagonal": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.0, 0.0), (-1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.0, 0.0), (1.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=135,
                                          speed=1.414,
                                          momentum=(-1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1)],
            init_sim=ExpectedSimState(momentum=(0, 2),
                                      kinetic_energy=2,
                                      relative_speed=2,
                                      distance=2,
                                      trajectories=BallTrajectories.DIVERGING),

            final_balls=[ExpectedBallState(angle=135,
                                           speed=1.414,
                                           velocity=(-1, 1),
                                           momentum=(-1, 1),
                                           kinetic_energy=1),
                         ExpectedBallState(angle=45,
                                           speed=1.414,
                                           velocity=(1, 1),
                                           momentum=(1, 1),
                                           kinetic_energy=1)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 2),
                                       kinetic_energy=2,
                                       relative_speed=2,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=None
        )
    ),
    "test04_elastic_collision_diagonal_opposing_symmetry": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, -1.5), (1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 1.5), (-1.0, -1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=-135,
                                          speed=1.414,
                                          momentum=(-1, -1),
                                          kinetic_energy=1)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=2,
                                      relative_speed=2*math.sqrt(2),
                                      distance=2*math.sqrt(2*1.5**2),
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=-135,
                                           speed=1.414,
                                           velocity=(-1, -1),
                                           momentum=(-1, -1),
                                           kinetic_energy=1),
                         ExpectedBallState(angle=45,
                                           speed=1.414,
                                           velocity=(1, 1),
                                           momentum=(1, 1),
                                           kinetic_energy=1)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=2,
                                       relative_speed=2*math.sqrt(2),
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1.15,
                                                 ball1_position=(-0.35, -0.35),
                                                 ball2_position=(0.35, 0.35)),
            intersect_info=None
        )
    ),
    "test05_elastic_intersection_right_triangle": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (0.0, 2.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=2,
                                          momentum=(0, 2),
                                          kinetic_energy=2)],
            init_sim=ExpectedSimState(momentum=(1, 3),
                                      kinetic_energy=3,
                                      relative_speed=math.sqrt(2),
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=45,
                                           speed=1.414,
                                           velocity=(1, 1),
                                           momentum=(1, 1),
                                           kinetic_energy=1),
                         ExpectedBallState(angle=90,
                                           speed=2,
                                           velocity=(0, 2),
                                           momentum=(0, 2),
                                           kinetic_energy=2)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(1, 3),
                                       kinetic_energy=3,
                                       relative_speed=math.sqrt(2),
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(1.5, 3.0),
                                                    ball1_time=3.0,
                                                    ball2_time=1.5)
        )
    ),
    "test06_elastic_collision_right_triangle": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (0.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=1,
                                          momentum=(0, 1),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(1, 2),
                                      kinetic_energy=1.5,
                                      relative_speed=1,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=90,
                                           speed=1,
                                           velocity=(0, 1),
                                           momentum=(0, 1),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=45,
                                           speed=1.414,
                                           velocity=(1, 1),
                                           momentum=(1, 1),
                                           kinetic_energy=1)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(1, 2),
                                       kinetic_energy=1.5,
                                       relative_speed=1,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=2.0,
                                                 ball1_position=(0.5, 2.0),
                                                 ball2_position=(1.5, 2.0)),
            intersect_info=None
        )
    ),
    "test07_elastic_intersection_small_angle_trajectories": (
        #                 mass,  position,               velocity
        PhysicsParameters(0.01, (-1.0, 0.0), (math.cos(45*DEGREES_TO_RADIANS),
                                              math.sin(45*DEGREES_TO_RADIANS))),
        #                 mass,  position,               velocity
        PhysicsParameters(0.01, (0.0, 0.0), (math.cos(46*DEGREES_TO_RADIANS),
                                             math.sin(46*DEGREES_TO_RADIANS))),
        # simulation time
        45.0,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.108,
                                          angle=45,
                                          speed=1,
                                          momentum=(0.00707, 0.00707),
                                          kinetic_energy=0.005),
                        ExpectedBallState(radius=0.108,
                                          angle=46,
                                          speed=1,
                                          momentum=(0.00695, 0.00719),
                                          kinetic_energy=0.005)],
            init_sim=ExpectedSimState(momentum=(0.01402, 0.01426),
                                      kinetic_energy=0.01,
                                      relative_speed=0.017,
                                      distance=1.0,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=45,
                                           speed=1,
                                           velocity=(0.707, 0.707),
                                           momentum=(0.00707, 0.00707),
                                           kinetic_energy=0.005),
                         ExpectedBallState(angle=46,
                                           speed=1,
                                           velocity=(0.695, 0.719),
                                           momentum=(0.00695, 0.00719),
                                           kinetic_energy=0.005)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0.01402, 0.01426),
                                       kinetic_energy=0.01,
                                       relative_speed=0.017,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(28.145, 29.145),
                                                    ball1_time=41.217,
                                                    ball2_time=40.516)
        )
    ),
    "test08_elastic_miss_parallel_same_direction": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 2.0), (1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(2, 0),
                                      kinetic_energy=1,
                                      relative_speed=0,
                                      distance=2,
                                      trajectories=BallTrajectories.CONSTANT),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(2, 0),
                                       kinetic_energy=1,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.CONSTANT,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=None
        )
    ),
    "test09_elastic_miss_parallel_opposing": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, -2.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 2.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=180,
                                          speed=1,
                                          momentum=(-1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=1,
                                      relative_speed=2,
                                      distance=4,
                                      trajectories=BallTrajectories.DIVERGING),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=180,
                                           speed=1,
                                           velocity=(-1, 0),
                                           momentum=(-1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=1,
                                       relative_speed=2,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=None
        )
    ),
    "test10_elastic_miss_parallel_velocity_differential": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 2.0), (2.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=2,
                                          momentum=(2, 0),
                                          kinetic_energy=2)],
            init_sim=ExpectedSimState(momentum=(3, 0),
                                      kinetic_energy=2.5,
                                      relative_speed=1,
                                      distance=2,
                                      trajectories=BallTrajectories.DIVERGING),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=0,
                                           speed=2,
                                           velocity=(2, 0),
                                           momentum=(2, 0),
                                           kinetic_energy=2)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(3, 0),
                                       kinetic_energy=2.5,
                                       relative_speed=1,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=None
        )
    ),
    "test11_elastic_collision_overtaking": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.0, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-2.5, 0.0), (2.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=2,
                                          momentum=(2, 0),
                                          kinetic_energy=2)],
            init_sim=ExpectedSimState(momentum=(3, 0),
                                      kinetic_energy=2.5,
                                      relative_speed=1,
                                      distance=2.5,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=2,
                                           velocity=(2, 0),
                                           momentum=(2, 0),
                                           kinetic_energy=2),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(3, 0),
                                       kinetic_energy=2.5,
                                       relative_speed=1,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1.5,
                                                 ball1_position=(1.5, 0.0),
                                                 ball2_position=(0.5, 0.0)),
            intersect_info=None
        )
    ),
    "test12_elastic_collision_different_masses": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-5.0, 0.0), (2.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(4.0, (5.0, 0.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=2,
                                          momentum=(2, 0),
                                          kinetic_energy=2),
                        ExpectedBallState(radius=0.794,
                                          angle=180,
                                          speed=1,
                                          momentum=(-4, 0),
                                          kinetic_energy=2)],
            init_sim=ExpectedSimState(momentum=(-2, 0),
                                      kinetic_energy=4,
                                      relative_speed=3,
                                      distance=10,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=180,
                                           speed=2.8,
                                           velocity=(-2.8, 0),
                                           momentum=(-2.8, 0),
                                           kinetic_energy=3.92),
                         ExpectedBallState(angle=0,
                                           speed=0.2,
                                           velocity=(0.2, 0),
                                           momentum=(0.8, 0),
                                           kinetic_energy=0.08)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(-2, 0),
                                       kinetic_energy=4,
                                       relative_speed=3,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=2.91,
                                                 ball1_position=(0.82, 0.0),
                                                 ball2_position=(2.09, 0.0)),
            intersect_info=None
        )
    ),
    "test13_elastic_collision_stationary_target": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (0.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0)],
            init_sim=ExpectedSimState(momentum=(1, 0),
                                      kinetic_energy=0.5,
                                      relative_speed=1,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=0,
                                           velocity=(0, 0),
                                           momentum=(0, 0),
                                           kinetic_energy=0),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(1, 0),
                                       kinetic_energy=0.5,
                                       relative_speed=1,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=2.0,
                                                 ball1_position=(0.5, 0.0),
                                                 ball2_position=(1.5, 0.0)),
            intersect_info=None
        )
    ),
    "test14_elastic_miss_dual_stationary_objects": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (0.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (0.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=0,
                                      relative_speed=0,
                                      distance=3,
                                      trajectories=BallTrajectories.CONSTANT),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=0,
                                           velocity=(0, 0),
                                           momentum=(0, 0),
                                           kinetic_energy=0),
                         ExpectedBallState(angle=0,
                                           speed=0,
                                           velocity=(0, 0),
                                           momentum=(0, 0),
                                           kinetic_energy=0)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=0,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.CONSTANT,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=None
        )
    ),
    "test15_elastic_collision_same_position_opposing_velocities": (
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=180,
                                          speed=1,
                                          momentum=(-1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=1,
                                      relative_speed=2,
                                      distance=0,
                                      trajectories=BallTrajectories.DIVERGING),

            final_balls=[ExpectedBallState(angle=180,
                                           speed=1,
                                           velocity=(-1, 0),
                                           momentum=(-1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=1,
                                       relative_speed=2,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=0.0,
                                                 ball1_position=(1.5, 0.0),
                                                 ball2_position=(1.5, 0.0)),
            intersect_info=None
        )
    ),
    "test16_elastic_collision_glancing_blow": (
        # mass, position, velocity
        PhysicsParameters(1.0, (3.0, 4.0), (-1.0, -0.5)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-3.3, -4.0), (0.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=-153.435,
                                          speed=1.118,
                                          momentum=(-1, -0.5),
                                          kinetic_energy=0.625),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=1,
                                          momentum=(0, 1),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(-1, 0.5),
                                      kinetic_energy=1.125,
                                      relative_speed=1.803,
                                      distance=10.183,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=-83.624,
                                           speed=0.466,
                                           velocity=(0.052, -0.463),
                                           momentum=(0.052, -0.463),
                                           kinetic_energy=0.108),
                         ExpectedBallState(angle=137.527,
                                           speed=1.425,
                                           velocity=(-1.052, 0.963),
                                           momentum=(-1.052, 0.963),
                                           kinetic_energy=1.017)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(-1, 0.5),
                                       kinetic_energy=1.125,
                                       relative_speed=1.803,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=5.31,
                                                 ball1_position=(-2.31, 1.345),
                                                 ball2_position=(-3.3, 1.31)),
            intersect_info=None
        )
    ),
    "test17_elastic_collision_oblique_angle": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.4, 3.0), (0.0, -2.0)),
        # mass, position, velocity
        PhysicsParameters(2.0, (-0.5, -3.0), (0.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=-90,
                                          speed=2,
                                          momentum=(0, -2),
                                          kinetic_energy=2),
                        ExpectedBallState(radius=0.63,
                                          angle=90,
                                          speed=1,
                                          momentum=(0, 2),
                                          kinetic_energy=1)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=3,
                                      relative_speed=3,
                                      distance=6.067,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=-17.492,
                                           speed=2,
                                           velocity=(1.9075, -0.601),
                                           momentum=(1.9075, -0.601),
                                           kinetic_energy=2),
                         ExpectedBallState(angle=162.508,
                                           speed=1,
                                           velocity=(-0.9538, 0.3006),
                                           momentum=(-1.9075, 0.601),
                                           kinetic_energy=1)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=3,
                                       relative_speed=3,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1.78,
                                                 ball1_position=(0.40, -0.56),
                                                 ball2_position=(-0.50, -1.22)),
            intersect_info=None
        )
    ),
    "test18_elastic_intersection_diagonal_trajectories": (
        # mass, position, velocity
        PhysicsParameters(1.0, (3.0, 4.0), (-1.0, -1.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-3.3, -4.0), (0.0, 1.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=-135,
                                          speed=1.414,
                                          momentum=(-1, -1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=1,
                                          momentum=(0, 1),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(-1, 0),
                                      kinetic_energy=1.5,
                                      relative_speed=2.236,
                                      distance=10.183,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=-135,
                                           speed=1.414,
                                           velocity=(-1, -1),
                                           momentum=(-1, -1),
                                           kinetic_energy=1),
                         ExpectedBallState(angle=90,
                                           speed=1,
                                           velocity=(0, 1),
                                           momentum=(0, 1),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(-1, 0),
                                       kinetic_energy=1.5,
                                       relative_speed=2.236,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=None,
            intersect_info=ExpectedIntersectionInfo(position=(-3.3, -2.3),
                                                    ball1_time=6.3,
                                                    ball2_time=1.7)
        )
    ),
    "test19_elastic_collision_side_pocket_pool_shot": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.495, 0.0), (0.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-0.495, -10.0), (0.0, 5.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=5,
                                          momentum=(0, 5),
                                          kinetic_energy=12.5)],
            init_sim=ExpectedSimState(momentum=(0, 5),
                                      kinetic_energy=12.5,
                                      relative_speed=5,
                                      distance=10.049,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=[ExpectedBallState(angle=5.768,
                                           speed=0.502,
                                           velocity=(0.5, 0.0505),
                                           momentum=(0.5, 0.0505),
                                           kinetic_energy=0.126),
                         ExpectedBallState(angle=95.768,
                                           speed=4.974,
                                           velocity=(-0.5, 4.9495),
                                           momentum=(-0.5, 4.9495),
                                           kinetic_energy=12.374)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(0, 5),
                                       kinetic_energy=12.5,
                                       relative_speed=5,
                                       trajectories=BallTrajectories.DIVERGING,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=1.98,
                                                 ball1_position=(0.495, 0.0),
                                                 ball2_position=(-0.495, -0.1)),
            intersect_info=None
        )
    ),
    "test20_inelastic_collision_head_on_symmetry": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=180,
                                          speed=1,
                                          momentum=(-1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(0, 0),
                                      kinetic_energy=1,
                                      relative_speed=2,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.63,
                                          angle=0,
                                          speed=0,
                                          velocity=(0, 0),
                                          momentum=(0, 0),
                                          kinetic_energy=0),
            final_sim=ExpectedSimState(momentum=(0, 0),
                                       kinetic_energy=0,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=1),

            collision_info=ExpectedCollisionInfo(time=1.0,
                                                 merged_position=(0, 0)),
            intersect_info=None
        )
    ),
    "test21_inelastic_collision_different_masses": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-3.0, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(2.0, (3.0, 0.0), (-1.0, 0.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.63,
                                          angle=180,
                                          speed=1,
                                          momentum=(-2, 0),
                                          kinetic_energy=1)],
            init_sim=ExpectedSimState(momentum=(-1, 0),
                                      kinetic_energy=1.5,
                                      relative_speed=2,
                                      distance=6,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.721,
                                          angle=180,
                                          speed=0.333,
                                          velocity=(-0.333, 0),
                                          momentum=(-1, 0),
                                          kinetic_energy=0.167),
            final_sim=ExpectedSimState(momentum=(-1, 0),
                                       kinetic_energy=0.167,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=1.333),

            collision_info=ExpectedCollisionInfo(time=2.44,
                                                 merged_position=(0.187, 0)),
            intersect_info=None
        )
    ),
    "test22_inelastic_collision_glancing_blow": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.4, 3.0), (0.0, -2.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-0.5, -3.0), (0.0, 1.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=-90,
                                          speed=2,
                                          momentum=(0, -2),
                                          kinetic_energy=2),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=1,
                                          momentum=(0, 1),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(0, -1),
                                      kinetic_energy=2.5,
                                      relative_speed=3,
                                      distance=6.067,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.63,
                                          angle=-90,
                                          speed=0.5,
                                          velocity=(0, -0.5),
                                          momentum=(0, -1),
                                          kinetic_energy=0.25),
            final_sim=ExpectedSimState(momentum=(0, -1),
                                       kinetic_energy=0.25,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=2.25),

            collision_info=ExpectedCollisionInfo(time=1.86,
                                                 merged_position=(-0.05, -0.93)),
            intersect_info=None
        )
    ),
    "test23_inelastic_collision_stationary_target": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (0.0, 0.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0)],
            init_sim=ExpectedSimState(momentum=(1, 0),
                                      kinetic_energy=0.5,
                                      relative_speed=1,
                                      distance=3,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.63,
                                          angle=0,
                                          speed=0.5,
                                          velocity=(0.5, 0),
                                          momentum=(1, 0),
                                          kinetic_energy=0.25),
            final_sim=ExpectedSimState(momentum=(1, 0),
                                       kinetic_energy=0.25,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=0.25),

            collision_info=ExpectedCollisionInfo(time=2.0,
                                                 merged_position=(1.0, 0)),
            intersect_info=None
        )
    ),
    "test24_inelastic_collision_side_pocket_pool_shot": (
        # mass, position, velocity
        PhysicsParameters(1.0, (0.495, 0.0), (0.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (-0.495, -10.0), (0.0, 5.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=0,
                                          momentum=(0, 0),
                                          kinetic_energy=0),
                        ExpectedBallState(radius=0.5,
                                          angle=90,
                                          speed=5,
                                          momentum=(0, 5),
                                          kinetic_energy=12.5)],
            init_sim=ExpectedSimState(momentum=(0, 5),
                                      kinetic_energy=12.5,
                                      relative_speed=5,
                                      distance=10.049,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.63,
                                          angle=90,
                                          speed=2.5,
                                          velocity=(0, 2.5),
                                          momentum=(0, 5),
                                          kinetic_energy=6.25),
            final_sim=ExpectedSimState(momentum=(0, 5),
                                       kinetic_energy=6.25,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=6.25),

            collision_info=ExpectedCollisionInfo(time=1.98,
                                                 merged_position=(0, -0.05)),
            intersect_info=None
        )
    ),
    "test25_inelastic_collision_diagonal_approach": (
        # mass, position, velocity
        PhysicsParameters(1.0, (-5.0, -5.0), (1.0, 1.0)),
        # mass, position, velocity
        PhysicsParameters(2.0, (5.0, 5.0), (-2.0, -2.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=45,
                                          speed=1.414,
                                          momentum=(1, 1),
                                          kinetic_energy=1),
                        ExpectedBallState(radius=0.63,
                                          angle=-135,
                                          speed=2.828,
                                          momentum=(-4, -4),
                                          kinetic_energy=8)],
            init_sim=ExpectedSimState(momentum=(-3, -3),
                                      kinetic_energy=9,
                                      relative_speed=4.243,
                                      distance=14.142,
                                      trajectories=BallTrajectories.CONVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.721,
                                          angle=-135,
                                          speed=1.414,
                                          velocity=(-1, -1),
                                          momentum=(-3, -3),
                                          kinetic_energy=3),
            final_sim=ExpectedSimState(momentum=(-3, -3),
                                       kinetic_energy=3,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=6),

            collision_info=ExpectedCollisionInfo(time=3.07,
                                                 merged_position=(-1.403, -1.403)),
            intersect_info=None
        )
    ),
    "test26_inelastic_collision_same_position_different_velocities": (
        # mass, position, velocity
        PhysicsParameters(1.0, (2.0, 0.0), (2.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (2.0, 0.0), (-4.0, 0.0)),
        # simulation time
        10,
        CollisionType.INELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=2,
                                          momentum=(2, 0),
                                          kinetic_energy=2),
                        ExpectedBallState(radius=0.5,
                                          angle=180,
                                          speed=4,
                                          momentum=(-4, 0),
                                          kinetic_energy=8)],
            init_sim=ExpectedSimState(momentum=(-2, 0),
                                      kinetic_energy=10,
                                      relative_speed=6,
                                      distance=0,
                                      trajectories=BallTrajectories.DIVERGING),

            final_balls=None,
            merged_ball=ExpectedBallState(radius=0.63,
                                          angle=180,
                                          speed=1,
                                          velocity=(-1, 0),
                                          momentum=(-2, 0),
                                          kinetic_energy=1),
            final_sim=ExpectedSimState(momentum=(-2, 0),
                                       kinetic_energy=1,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.MERGED,
                                       ke_lost=9),

            collision_info=ExpectedCollisionInfo(time=0.0,
                                                 merged_position=(2.0, 0)),
            intersect_info=None
        )
    ),
    "test27_elastic_collision_same_position_same_velocity": (
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (1.0, 0.0)),
        # mass, position, velocity
        PhysicsParameters(1.0, (1.5, 0.0), (1.0, 0.0)),
        # simulation time
        10,
        CollisionType.ELASTIC,
        ExpectedResults(
            init_balls=[ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5),
                        ExpectedBallState(radius=0.5,
                                          angle=0,
                                          speed=1,
                                          momentum=(1, 0),
                                          kinetic_energy=0.5)],
            init_sim=ExpectedSimState(momentum=(2, 0),
                                      kinetic_energy=1,
                                      relative_speed=0,
                                      distance=0,
                                      trajectories=BallTrajectories.CONSTANT),

            final_balls=[ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5),
                         ExpectedBallState(angle=0,
                                           speed=1,
                                           velocity=(1, 0),
                                           momentum=(1, 0),
                                           kinetic_energy=0.5)],
            merged_ball=None,
            final_sim=ExpectedSimState(momentum=(2, 0),
                                       kinetic_energy=1,
                                       relative_speed=0,
                                       trajectories=BallTrajectories.CONSTANT,
                                       ke_lost=0),

            collision_info=ExpectedCollisionInfo(time=0.0,
                                                 ball1_position=(1.5, 0.0),
                                                 ball2_position=(1.5, 0.0)),
            intersect_info=None
        )
    ),
}
