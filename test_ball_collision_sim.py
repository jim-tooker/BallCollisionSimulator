#!/usr/bin/env python
"""
This module contains unit tests for the BallCollisionSimulator class.

It uses pytest to run various test scenarios for ball collisions, 
including elastic and inelastic collisions, intersections, and misses.
"""

from typing import Tuple, List, Dict, Union
import sys
import argparse
import math
import re
import pytest
import vpython as vp
from ball_collision_sim import PhysicsParameters, BallCollisionSimulator, CollisionType
from test_cases import TESTS, ACTIVE_TESTS, ExpectedResults, ExpectedBallState

__author__ = "Jim Tooker"


def get_active_tests() -> List[Tuple[str, Tuple[List[PhysicsParameters],
                                               float,
                                               CollisionType,
                                               ExpectedResults]]]:
    """
    Get the list of active tests to run and returns a list of active tests with their parameters.

    Returns:
        List[Tuple[str, Tuple[List[PhysicsParameters], float, CollisionType,  ExpectedResults]]]  
    """
    if "all" in ACTIVE_TESTS:
        return [(name, params) for name, params in TESTS.items()]  # All tests

    return [(name, TESTS[name]) for name in ACTIVE_TESTS if name in TESTS]  # Specific tests


def get_characteristics(test_name: str) -> Dict[str, Union[bool, str]]:
    """
    Extract test characteristics from the test name.

    Args:
        test_name (str): The name of the test.

    Returns:
        Dict[str, Union[bool, str]]: A dictionary containing test characteristics.
    """
    # Define regular expressions for each category
    elastic_re = re.compile(r'_elastic_')
    inelastic_re = re.compile(r'_inelastic_')
    collision_re = re.compile(r'_collision_')
    intersection_re = re.compile(r'_intersection_')
    miss_re = re.compile(r'_miss_')

    # Compute characteristics
    is_elastic = elastic_re.search(test_name) is not None
    is_inelastic = inelastic_re.search(test_name) is not None
    is_collision = collision_re.search(test_name) is not None
    is_intersection = intersection_re.search(test_name) is not None
    is_miss = miss_re.search(test_name) is not None

    # Check validity
    is_valid = (is_elastic != is_inelastic) and (sum([is_collision, is_intersection, is_miss]) == 1)

    # Create a human-readable test description
    collision_type = "Elastic" if is_elastic else "Inelastic" if is_inelastic else "Unknown"
    event_type = (", With Collision" if is_collision else 
                  ", With Intersection" if is_intersection else 
                  ", With Neither Collision nor Intersection" if is_miss else ", Unknown")
    description = f"{collision_type}{event_type}"

    return {
        'elastic': is_elastic,
        'inelastic': is_inelastic,
        'collision': is_collision,
        'intersection': is_intersection,
        'miss': is_miss,
        'is_valid': is_valid,
        'description': description
    }


def value_approx(v1: float, v2: float) -> bool:
    """
    Helper function for approximate value comparison.

    Args:
        v1 (float): First value to compare.
        v2 (float): Second value to compare.

    Returns:
        bool: True if the values are approximately equal, False otherwise.
    """
    return pytest.approx(v1, abs=0.001) == v2


def vector_approx(v1: vp.vector, v2: Union[Tuple[float, float], vp.vector]) -> bool:
    """
    Helper function for approximate vector comparison.

    Args:
        v1 (vp.vector): First vector to compare.
        v2 (Union[Tuple[float, float], vp.vector]): Second vector to compare.

    Returns:
        bool: True if the vectors are approximately equal, False otherwise.
    """
    if not isinstance(v2, vp.vector):
        v2 = vp.vector(*v2, 0)

    return pytest.approx((v1 - v2).mag, abs=0.001) == 0


def get_radius(mass: float) -> float:
    """
    Calculate the radius based on mass given.
    In this sim, radius is proportional to the mass (0.5m for 1 kg)

    Args:
    mass (float): Mass of the ball (kg).

    Returns:
        float: The radius of the ball (m).
    """
    return float(0.5 * (mass ** (1/3)))


def get_angle(velocity: Union[Tuple[float, float], vp.vector]) -> float:
    """
    Calculate the angle of the ball's velocity vector.

    Args:
    velocity (Union[Tuple[float, float], vp.vector]): Velocity vector of the ball.

    Returns:
        float: The angle of the ball (degrees).
    """
    if not isinstance(velocity, vp.vector):
        velocity = vp.vector(*velocity, 0)

    return float(math.degrees(math.atan2(velocity.y, velocity.x)))


def get_speed(velocity: Union[Tuple[float, float], vp.vector]) -> float:
    """
    Calculate the speed of the ball.

    Args:
    velocity (Union[Tuple[float, float], vp.vector]): Velocity vector of the ball.

    Returns:
        float: The speed of the ball (m/s).
    """
    if not isinstance(velocity, vp.vector):
        velocity = vp.vector(*velocity, 0)

    return float(vp.mag(velocity))


def get_momentum(mass: float, velocity: Union[Tuple[float, float], vp.vector]) -> vp.vector:
    """
    Calculate the momentum vector of the ball.

    Args:
    mass (float): Mass of the ball (kg).
    velocity (Union[Tuple[float, float], vp.vector]): Velocity vector of the ball.

    Returns:
        vp.vector: The momentum vector of the ball.
    """
    if not isinstance(velocity, vp.vector):
        velocity = vp.vector(*velocity, 0)

    return velocity * mass


def get_kinetic_energy(mass: float, speed: float) -> float:
    """
    Calculate the kinetic energy of the ball.

    Args:
    mass (float): Mass of the ball (kg).
    speed (float): The speed of the ball (m/s).

    Returns:
        float: The kinetic energy of the ball (J).
    """
    return 0.5 * mass * (speed**2)


def get_relative_speed(v1: Union[Tuple[float, float], vp.vector],
                       v2: Union[Tuple[float, float], vp.vector]) -> float:
    """
    Calculate the relative speed of the two balls with respect to each other.

    Args:
    v1 (Union[Tuple[float, float], vp.vector]): Velocity vector of ball 1.
    v2 (Union[Tuple[float, float], vp.vector]): Velocity vector of ball 2.

    Returns:
        float: The relative speed of the balls with respect to each other (m/s).
    """
    if not isinstance(v1, vp.vector):
        v1 = vp.vector(*v1, 0)

    if not isinstance(v2, vp.vector):
        v2 = vp.vector(*v2, 0)

    return float(vp.mag(v1 - v2))


def get_distance(p1: vp.vector, p2: vp.vector) -> float:
    """
    Calculate the distance between the two balls.

    Args:
    v1 (vp.vector): Velocity vector of ball 1.
    v2 (vp.vector): Velocity vector of ball 2.

    Returns:
        float: The distance between the balls (m).
    """
    return float(vp.mag(p1 - p2))


@pytest.mark.parametrize("ball_params, sim_time, collision_type, expected, test_name",
                         [(*test[1], test[0]) for test in get_active_tests()],
                         ids=[test[0] for test in get_active_tests()])
def test_ball_collision(ball_params: List[PhysicsParameters],
                        sim_time: float,
                        collision_type: CollisionType,
                        expected: ExpectedResults,
                        test_name: str) -> None:
    """
    Test function for ball collision scenarios.

    This function runs a ball collision simulation with given parameters and compares
    the results against expected values.

    Args:
        ball_params (List[PhysicsParameters]): List of parameters for the balls.
        sim_time (float): Simulation time.
        collision_type (CollisionType): Type of collision (elastic or inelastic).
        expected (ExpectedResults): Expected results of the simulation.
        test_name (str): Name of the test case.
    """
    test_characteristics = get_characteristics(test_name)

    # Print the test description
    print(f"\n\nRunning Test: {test_name} -> Test Type: {test_characteristics['description']}")

    # Ensure the test name is valid
    assert test_characteristics['is_valid'], f"Test name '{test_name}' does not contain valid characterizations."

    # Create and run the simulation
    sim = BallCollisionSimulator.create_simulator(ball_params,
                                                  sim_time,
                                                  collision_type)
    sim.run()

    # Check for None object values
    if test_characteristics['collision']:
        assert sim.collision_info
    else:
        assert sim.collision_info is None
    if test_characteristics['intersection']:
        assert sim.intersect_info
    else:
        assert sim.intersect_info is None
    if test_characteristics['inelastic'] and test_characteristics['collision']:
        assert sim.merged_ball
    else:
        assert sim.merged_ball is None

    # Test initial states for balls
    init_expected_ball_speed: List[float] = []
    init_expected_ball_momentum: List[vp.vector] = []
    init_expected_ball_ke: List[float] = []
    for i, ball in enumerate(sim.initial.balls):
        assert value_approx(ball.radius, get_radius(ball_params[i].mass))
        assert value_approx(ball.angle, get_angle(ball_params[i].velocity))
        assert vector_approx(ball.velocity, ball_params[i].velocity)
        init_expected_ball_speed.append(get_speed(ball_params[i].velocity))
        assert value_approx(ball.speed, init_expected_ball_speed[i])
        init_expected_ball_momentum.append(get_momentum(ball_params[i].mass, ball_params[i].velocity))
        assert vector_approx(ball.momentum, init_expected_ball_momentum[i])
        init_expected_ball_ke.append(get_kinetic_energy(ball_params[i].mass, init_expected_ball_speed[i]))
        assert value_approx(ball.kinetic_energy, init_expected_ball_ke[i])

    # Test initial states for sim
    init_expected_total_momentum: vp.vector = init_expected_ball_momentum[0] + init_expected_ball_momentum[1]
    assert vector_approx(sim.initial.momentum, init_expected_total_momentum)
    init_expected_total_ke: float = init_expected_ball_ke[0] + init_expected_ball_ke[1]
    assert value_approx(sim.initial.kinetic_energy, init_expected_total_ke)
    assert value_approx(sim.initial.relative_speed, get_relative_speed(ball_params[0].velocity,
                                                                       ball_params[1].velocity))
    assert value_approx(sim.initial.distance, get_distance(ball_params[0].position,
                                                           ball_params[1].position))
    assert sim.initial.trajectories == expected.init_sim.trajectories

    # If no collision, then ball velocities don't change (just get them from params values)
    if not test_characteristics['collision']:
        expected.final_balls = [ExpectedBallState(), ExpectedBallState()]
        expected.final_balls[0].velocity = (ball_params[0].velocity.x, ball_params[0].velocity.y)
        expected.final_balls[1].velocity = (ball_params[1].velocity.x, ball_params[1].velocity.y)

    # Test final states for balls
    final_expected_ball_speed: List[float] = []
    final_expected_ball_momentum: List[vp.vector] = []
    final_expected_ball_ke: List[float] = []
    if sim.merged_ball:
        assert expected.merged_ball
        merged_ball_mass: float = ball_params[0].mass + ball_params[1].mass
        assert value_approx(sim.merged_ball.radius, get_radius(merged_ball_mass))
        assert expected.merged_ball.velocity
        assert value_approx(sim.merged_ball.angle, get_angle(expected.merged_ball.velocity))
        assert vector_approx(sim.merged_ball.velocity, expected.merged_ball.velocity)
        final_expected_ball_speed.append(get_speed(expected.merged_ball.velocity))
        assert value_approx(sim.merged_ball.speed, final_expected_ball_speed[0])
        final_expected_ball_momentum.append(get_momentum(merged_ball_mass, expected.merged_ball.velocity))
        assert vector_approx(sim.merged_ball.momentum, final_expected_ball_momentum[0])
        final_expected_ball_ke.append(get_kinetic_energy(merged_ball_mass, final_expected_ball_speed[0]))
        assert value_approx(sim.merged_ball.kinetic_energy, final_expected_ball_ke[0])
    else:  # Else, elastic collision, no merged ball
        assert sim.balls
        assert expected.final_balls
        for i, (ball, expected_final_ball) in enumerate(zip(sim.balls, expected.final_balls)):
            assert value_approx(ball.radius, get_radius(ball_params[i].mass))
            assert expected_final_ball.velocity
            assert value_approx(ball.angle, get_angle(expected_final_ball.velocity))
            assert vector_approx(ball.velocity, expected_final_ball.velocity)
            final_expected_ball_speed.append(get_speed(expected_final_ball.velocity))
            assert value_approx(ball.speed, final_expected_ball_speed[i])
            final_expected_ball_momentum.append(get_momentum(ball_params[i].mass, expected_final_ball.velocity))
            assert vector_approx(ball.momentum, final_expected_ball_momentum[i])
            final_expected_ball_ke.append(get_kinetic_energy(ball_params[i].mass, final_expected_ball_speed[i]))
            assert value_approx(ball.kinetic_energy, final_expected_ball_ke[i])

    # Calculate final momentum
    final_expected_total_momentum: vp.vector = vp.vector(0,0,0)
    for momentum in final_expected_ball_momentum:
        final_expected_total_momentum += momentum

    # Calculate final KE
    final_expected_total_ke: float = 0.0
    for ke in final_expected_ball_ke:
        final_expected_total_ke += ke

    # Test final states for sim
    assert vector_approx(sim.momentum, final_expected_total_momentum)
    assert value_approx(sim.kinetic_energy, final_expected_total_ke)
    if sim.merged_ball:
        assert value_approx(sim.relative_speed, 0.0)
    else:
        assert expected.final_balls
        assert expected.final_balls[0].velocity
        assert expected.final_balls[1].velocity
        assert value_approx(sim.relative_speed, get_relative_speed(expected.final_balls[0].velocity,
                                                                   expected.final_balls[1].velocity))
    assert sim.trajectories == expected.final_sim.trajectories

    # Check for conservation of momentum
    assert vector_approx(init_expected_total_momentum, sim.momentum)

    # Check KE lost (for inelastic)
    if sim.merged_ball:
        assert expected.final_sim.ke_lost
        assert value_approx(sim.ke_lost, expected.final_sim.ke_lost)
    # Else, check for conservation of KE
    else:
        assert value_approx(init_expected_total_ke, sim.kinetic_energy)
        assert value_approx(sim.ke_lost, 0)

    # Test collision/intersection info
    if sim.collision_info:
        assert expected.collision_info
        assert value_approx(sim.collision_info.time, expected.collision_info.time)
        if sim.merged_ball:
            assert sim.collision_info.merged_ball
            assert expected.collision_info.merged_position
            assert vector_approx(sim.collision_info.merged_ball.position, expected.collision_info.merged_position)
        else:
            assert expected.collision_info.ball1_position
            assert expected.collision_info.ball2_position
            assert vector_approx(sim.collision_info.ball1.position, expected.collision_info.ball1_position)
            assert vector_approx(sim.collision_info.ball2.position, expected.collision_info.ball2_position)
    elif sim.intersect_info:
        assert expected.intersect_info
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
        sys.exit(result)
    else:
        BallCollisionSimulator.quit_simulation()
