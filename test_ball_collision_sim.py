#!/usr/bin/env python
"""
This module contains unit tests for the BallCollisionSimulator class.

It uses pytest to run various test scenarios for ball collisions, 
including elastic and inelastic collisions, intersections, and misses.
"""

from typing import Tuple, List, Dict, Union, Optional
import sys
import argparse
import re
import pytest
import vpython as vp
from ball_collision_sim import PhysicsParameters, BallCollisionSimulator, CollisionType
from test_cases import TESTS, ACTIVE_TESTS, ExpectedResults

__author__ = "Jim Tooker"


def get_active_tests() -> List[Tuple[str,
                                     Tuple[PhysicsParameters,
                                           PhysicsParameters,
                                           float,
                                           CollisionType,
                                           ExpectedResults]]]:
    """
    Get the list of active tests to run.

    Returns:
        List[Tuple[str, Tuple[PhysicsParameters, PhysicsParameters, float, CollisionType, ExpectedResults]]]: 
        A list of active tests with their parameters.
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


def value_approx(v1: Optional[float], v2: Optional[float]) -> bool:
    """
    Helper function for approximate value comparison.

    Args:
        v1 (Optional[float]): First value to compare.
        v2 (Optional[float]): Second value to compare.

    Returns:
        bool: True if the values are approximately equal, False otherwise.
    """
    return pytest.approx(v1, abs=0.001) == v2


def vector_approx(v1: vp.vector, v2: Optional[Union[Tuple[float, float], vp.vector]]) -> bool:
    """
    Helper function for approximate vector comparison.

    Args:
        v1 (vp.vector): First vector to compare.
        v2 (Optional[Union[Tuple[float, float], vp.vector]]): Second vector to compare.

    Returns:
        bool: True if the vectors are approximately equal, False otherwise.
    """
    return pytest.approx((v1 - vp.vector(*v2, 0)).mag, abs=0.001) == 0.0


@pytest.mark.parametrize("ball1_params, ball2_params, sim_time, collision_type, expected, test_name",
                         [(*test[1], test[0]) for test in get_active_tests()],
                         ids=[test[0] for test in get_active_tests()])
def test_ball_collision(ball1_params: PhysicsParameters,
                        ball2_params: PhysicsParameters,
                        sim_time: float,
                        collision_type: CollisionType,
                        expected: ExpectedResults,
                        test_name: str) -> None:
    """
    Test function for ball collision scenarios.

    This function runs a ball collision simulation with given parameters and compares
    the results against expected values.

    Args:
        ball1_params (PhysicsParameters): Parameters for the first ball.
        ball2_params (PhysicsParameters): Parameters for the second ball.
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
        assert expected.merged_ball
        assert value_approx(sim.merged_ball.angle, expected.merged_ball.angle)
        assert value_approx(sim.merged_ball.speed, expected.merged_ball.speed)
        assert vector_approx(sim.merged_ball.velocity, expected.merged_ball.velocity)
        assert vector_approx(sim.merged_ball.momentum, expected.merged_ball.momentum)
        assert value_approx(sim.merged_ball.kinetic_energy, expected.merged_ball.kinetic_energy)
    else:
        assert expected.final_balls
        for i, ball in enumerate(expected.final_balls):
            assert value_approx(sim.balls[i].angle, ball.angle)
            assert value_approx(sim.balls[i].speed, ball.speed)
            assert vector_approx(sim.balls[i].velocity, ball.velocity)
            assert vector_approx(sim.balls[i].momentum, ball.momentum)
            assert value_approx(sim.balls[i].kinetic_energy, ball.kinetic_energy)

    # Test final states for sim
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
    assert vector_approx(sim.momentum, expected.final_sim.momentum)
    assert value_approx(sim.kinetic_energy, expected.final_sim.kinetic_energy)
    assert value_approx(sim.relative_speed, expected.final_sim.relative_speed)
    assert sim.trajectories == expected.final_sim.trajectories
    if sim.merged_ball:
        assert value_approx(sim.ke_lost, expected.final_sim.ke_lost)

    # Test collision/intersection info
    if sim.collision_info:
        assert expected.collision_info
        assert value_approx(sim.collision_info.time, expected.collision_info.time)
        if sim.merged_ball:
            assert sim.collision_info.merged_ball
            assert vector_approx(sim.collision_info.merged_ball.position, expected.collision_info.merged_position)
        else:
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
