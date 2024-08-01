from typing import Tuple, List, Dict, Final, Optional
from dataclasses import dataclass, field
import sys
import argparse
import math
import pytest
import vpython as vp
from ball_collision_sim import PhysicsParameters, \
    BallCollisionSimulator, \
    CollisionType, \
    BallTrajectories

__author__ = "Jim Tooker"


DEGREES_TO_RADIANS = math.pi/180.0


@dataclass
class ExpectedBallState:
    radius: Optional[float] = field(default=None)
    angle: Optional[float] = field(default=None)
    velocity: Optional[Tuple[float, float]] = field(default=None)
    speed: Optional[float] = field(default=None)
    momentum: Optional[Tuple[float, float]] = field(default=None)
    kinetic_energy: Optional[float] = field(default=None)


@dataclass
class ExpectedSimState:
    momentum: Optional[Tuple[float, float]] = field(default=None)
    kinetic_energy: Optional[float] = field(default=None)
    relative_speed: Optional[float] = field(default=None)
    distance: Optional[float] = field(default=None)
    trajectories: Optional[BallTrajectories] = field(default=None)
    ke_lost: Optional[float] = field(default=None)


@dataclass
class ExpectedCollisionInfo:
    time: Optional[float] = field(default=None)
    ball1_position: Optional[Tuple[float, float]] = field(default=None)
    ball2_position: Optional[Tuple[float, float]] = field(default=None)
    merged_position: Optional[Tuple[float, float]] = field(default=None)


@dataclass
class ExpectedIntersectionInfo:
    position: Optional[Tuple[float, float]] = field(default=None)
    ball1_time: Optional[float] = field(default=None)
    ball2_time: Optional[float] = field(default=None)


@dataclass
class ExpectedResults:
    init_balls: List[ExpectedBallState] = field(default_factory=list)
    init_sim: ExpectedSimState = field(default_factory=ExpectedSimState)

    final_balls: Optional[List[ExpectedBallState]] = field(default_factory=list)
    merged_ball: Optional[ExpectedBallState] = field(default_factory=ExpectedBallState)
    final_sim: ExpectedSimState = field(default_factory=ExpectedSimState)

    collision_info: Optional[ExpectedCollisionInfo] = None
    intersect_info: Optional[ExpectedIntersectionInfo] = None


def value_approx(v1, v2) -> bool:
    """Helper function for value comparison"""
    return pytest.approx(v1, abs=0.001) == v2


def vector_approx(v1, v2) -> bool:
    """Helper function for vector comparison"""
    return pytest.approx((v1 - vp.vector(*v2, 0)).mag, abs=0.001) == 0.0


TESTS: Final[Dict] = {
    "test01_x_axis_only": (
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
    "test02_converging_balls": (
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
    "test03_diverging_balls": (
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
    "test04_from_sw_and_ne": (
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
    "test05_right_triangle_no_collision": (
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
    "test06_right_triangle_collision": (
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
    "test07_small_angle_converging": (
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
    "test08_parallel_balls_same_direction": (
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
    "test09_parallel_balls_diff_direction": (
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
    "test10_parallel_balls_same_direction_one_faster": (
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
    "test11_overtaking_ball": (
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
    "test12_different_masses_colliding": (
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
    "test13_one_object_not_moving": (
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
    "test14_both_objects_not_moving": (
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
    "test15_same_initial_position": (
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
    "test16_a_glancing_blow": (
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
    "test17_another_glancing_blow": (
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
    "test18_easy_intersection": (
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
    "test19_side_pocket_pool_shot": (
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
    "test20_inelastic_head_on_collision": (
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
    "test21_inelastic_different_masses": (
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
    "test22_inelastic_glancing_blow": (
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
    "test23_inelastic_one_object_not_moving": (
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
    "test24_inelastic_side_pocket_pool_shot": (
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
    "test25_inelastic_come_in_sideways": (
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
    "test26_inelastic_same_initial_position": (
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
    "test27_same_initial_position_same_velocity": (
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

ACTIVE_TESTS: Final[List[str]] = [
    "all",   # This will run everything
    # "test01_x_axis_only",
    # "test02_converging_balls",
    # "test03_diverging_balls",
    # "test04_from_sw_and_ne",
    # "test05_right_triangle_no_collision",
    # "test06_right_triangle_collision",
    # "test07_small_angle_converging",
    # "test08_parallel_balls_same_direction",
    # "test09_parallel_balls_diff_direction",
    # "test10_parallel_balls_same_direction_one_faster",
    # "test11_overtaking_ball",
    # "test12_different_masses_colliding",
    # "test13_one_object_not_moving",
    # "test14_both_objects_not_moving",
    # "test15_same_initial_position",
    # "test16_a_glancing_blow",
    # "test17_another_glancing_blow",
    # "test18_easy_intersection",
    # "test19_side_pocket_pool_shot",
    # "test20_inelastic_head_on_collision",
    # "test21_inelastic_different_masses",
    # "test22_inelastic_glancing_blow",
    # "test23_inelastic_one_object_not_moving",
    # "test24_inelastic_side_pocket_pool_shot",
    # "test25_inelastic_come_in_sideways",
    # "test26_inelastic_same_initial_position",
    # "test27_same_initial_position_same_velocity",
]


def get_active_tests() -> List[List]:
    if "all" in ACTIVE_TESTS:
        return list(TESTS.values())  # All tests
    # Specific tests
    return [TESTS[name] for name in ACTIVE_TESTS if name in TESTS]


@pytest.mark.parametrize("ball1_params, ball2_params, sim_time, collision_type, expected",
                         get_active_tests())
def test_ball_collision(ball1_params: List,
                        ball2_params: List,
                        sim_time: List,
                        collision_type: List,
                        expected: List):
    sim = BallCollisionSimulator.create_simulator(ball1_params,
                                                  ball2_params,
                                                  sim_time,
                                                  collision_type)
    sim.run()

    # Test initial states for balls
    for i, ball in enumerate(expected.init_balls):
        assert value_approx(sim.initial.balls[i].radius, ball.radius)
        assert value_approx(sim.initial.balls[i].angle, ball.angle)
        assert value_approx(sim.initial.balls[i].speed, ball.speed)
        assert vector_approx(sim.initial.balls[i].momentum, ball.momentum)
        assert value_approx(sim.initial.balls[i].kinetic_energy, ball.kinetic_energy)

    # Test initial states for sim
    assert vector_approx(sim.initial.momentum, expected.init_sim.momentum)
    assert value_approx(sim.initial.kinetic_energy, expected.init_sim.kinetic_energy)
    assert value_approx(sim.initial.relative_speed, expected.init_sim.relative_speed)
    assert value_approx(sim.initial.distance, expected.init_sim.distance)
    assert sim.initial.trajectories == expected.init_sim.trajectories

    # Test final states for balls
    if sim.merged_ball:
        assert value_approx(sim.merged_ball.angle, expected.merged_ball.angle)
        assert value_approx(sim.merged_ball.speed, expected.merged_ball.speed)
        assert vector_approx(sim.merged_ball.velocity, expected.merged_ball.velocity)
        assert vector_approx(sim.merged_ball.momentum, expected.merged_ball.momentum)
        assert value_approx(sim.merged_ball.kinetic_energy, expected.merged_ball.kinetic_energy)
    else:
        for i, ball in enumerate(expected.final_balls):
            assert value_approx(sim.balls[i].angle, ball.angle)
            assert value_approx(sim.balls[i].speed, ball.speed)
            assert vector_approx(sim.balls[i].velocity, ball.velocity)
            assert vector_approx(sim.balls[i].momentum, ball.momentum)
            assert value_approx(sim.balls[i].kinetic_energy, ball.kinetic_energy)

    # Test final states for sim
    assert vector_approx(sim.momentum, expected.final_sim.momentum)
    assert value_approx(sim.kinetic_energy, expected.final_sim.kinetic_energy)
    assert value_approx(sim.relative_speed, expected.final_sim.relative_speed)
    assert sim.trajectories == expected.final_sim.trajectories
    assert value_approx(sim.ke_lost, expected.final_sim.ke_lost)

    # Test collision/intersection info
    if sim.collision_info:
        assert value_approx(sim.collision_info.time, expected.collision_info.time)
        if sim.merged_ball:
            assert vector_approx(sim.collision_info.merged_ball.position, expected.collision_info.merged_position)
        else:
            assert vector_approx(sim.collision_info.ball1.position, expected.collision_info.ball1_position)
            assert vector_approx(sim.collision_info.ball2.position, expected.collision_info.ball2_position)
    elif sim.intersect_info:
        assert vector_approx(sim.intersect_info.position, expected.intersect_info.position)
        assert value_approx(sim.intersect_info.ball1_time, expected.intersect_info.ball1_time)
        assert value_approx(sim.intersect_info.ball2_time, expected.intersect_info.ball2_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ball Collision Simulator Tester')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    if args.no_gui:
        BallCollisionSimulator.disable_gui(True)

    result = pytest.main(["test_ball_collision_sim.py"])

    if args.no_gui:
        # exit with "not" return code from test
        sys.exit(result)
    else:
        # Quit Simulation
        BallCollisionSimulator.quit_simulation()
