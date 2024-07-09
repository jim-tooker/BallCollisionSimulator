import unittest
import sys
import argparse
import math
import vpython as vp
from BallCollisionSimulator import PhysicsParameters, BallCollisionSimulator

DEGREES_TO_RADIANS = math.pi/180.0

class TestBallCollisionSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = None

    def tearDown(self):
        if self.sim:
            self.sim = None

    def test01_x_axis_only(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0),  (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.5, 0.0), (-1.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                               # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 180.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 1.0)
        self.assertAlmostEqual(self.sim.ball1.angle, 180.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(-0.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(0.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test02_converging_lines(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0),  (1.0, 1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.5, 0.0), (-1.0, 1.0)),  # Ball 2: mass, position, velocity
            5.0                               # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 135.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 1.0)
        self.assertAlmostEqual(self.sim.ball1.angle, 135.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 45.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(-0.5, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(0.5, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(-1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test03_diverging_lines(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.0, 0.0),  (-1.0, 1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.0, 0.0),   (1.0, 1.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 2.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.0)
        self.assertGreater(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 135.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 45.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(-1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 2.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 135.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 45.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test04_from_sw_and_ne(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, -1.5),  (1.0, 1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.5,  1.5), (-1.0,-1.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 2*math.sqrt(2*1.5**2))
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2*math.sqrt(2))
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, -135.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-1.0, -1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 1.15)
        self.assertAlmostEqual(self.sim.ball1.angle, -135.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 45.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, -1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(-0.35, -0.35, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(0.35, 0.35, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(-1.0, -1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test05_right_triangle_no_collision(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0),  (1.0, 1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.5, 0.0),   (0.0, 2.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, math.sqrt(2))
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(1.0, 3.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(0.0, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.intersection_info.position - vp.vector(1.5, 3.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.intersection_info.ball1_time, 3.0)
        self.assertAlmostEqual(self.sim.intersection_info.ball2_time, 1.5)


    def test06_right_triangle_collision(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0),  (1.0, 1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0,  (1.5, 0.0),   (0.0, 1.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 1.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(1.0, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 2.0)
        self.assertAlmostEqual(self.sim.ball1.angle, 90.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 45.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(0.5, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(1.5, 2.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 1.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test07_small_angle_converging(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(0.01, (-1.0, 0.0),  (math.cos(45*DEGREES_TO_RADIANS),
                                                   math.sin(45*DEGREES_TO_RADIANS))),  # Ball 1: mass, position, velocity
            PhysicsParameters(0.01,  (0.0, 0.0),  (math.cos(46*DEGREES_TO_RADIANS),
                                                   math.sin(46*DEGREES_TO_RADIANS))),  # Ball 2: mass, position, velocity
            45.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 1.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 0.215, places=3)
        self.assertAlmostEqual(self.sim.relative_speed, 0.017, places=3)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 46.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(0.00707, 0.00707, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.00695, 0.00719, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.01402, 0.01426, 0.0)).mag, 0.0, places=3)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 45.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 46.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(0.00707, 0.00707, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(0.00695, 0.00719, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.intersection_info.position - vp.vector(28.145, 29.145, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual(self.sim.intersection_info.ball1_time, 41.217, places=3)
        self.assertAlmostEqual(self.sim.intersection_info.ball2_time, 40.516, places=3)


    def test08_parallel_lines_same_direction(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (0.0, 0.0),  (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (0.0, 2.0),  (1.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 2.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 0.0)
        self.assertEqual(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test09_parallel_lines_diff_direction(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (0.0, -2.0),  (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (0.0, 2.0),  (-1.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 4.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.0)
        self.assertEqual(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 180.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 180.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test10_parallel_lines_same_direction_one_faster(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (0.0, 0.0),  (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (0.0, 2.0),  (2.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                                # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 2.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 1.0)
        self.assertEqual(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(3.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test11_overtaking_parallel_lines(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (0.0, 0.0),  (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (-2.5, 0.0), (2.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                              # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 2.5)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 1.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(3.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 1.5)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(1.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(0.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test12_different_masses_colliding(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-5.0, 0.0), (2.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(4.0, (5.0, 0.0), (-1.0, 0.0)),  # Ball 2: mass, position, velocity
            10.0                             # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 10.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.294, places=3)
        self.assertAlmostEqual(self.sim.relative_speed, 3.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 180.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-4.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(-2.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 2.91)
        self.assertAlmostEqual(self.sim.ball1.angle, 180.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-2.8, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(0.8, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(0.82, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(2.09, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(-2.8, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(0.2, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)
            
        
    def test13_one_object_not_moving(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0), (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (1.5, 0.0),  (0.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                              # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 1.0)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 2.0)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(0.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(1.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)
            
        
    def test14_both_objects_not_moving(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (-1.5, 0.0), (0.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (1.5, 0.0),  (0.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                              # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 3.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 0.0)
        self.assertEqual(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test15_same_initial_position(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (1.5, 0.0), (1.0, 0.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (1.5, 0.0),  (-1.0, 0.0)),  # Ball 2: mass, position, velocity
            5.0                               # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 0.0)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.0)
        self.assertEqual(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, 0.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 180.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(0.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 0.0)
        self.assertAlmostEqual(self.sim.ball1.angle, 180.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 0.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(1.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(1.5, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)


    def test16_a_glancing_blow(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0,  (3.0, 4.0),  (-1.0, -0.5)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (-3.3, -4.0),  (0.0, 1.0)),   # Ball 2: mass, position, velocity
            10.0                                 # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 10.183, places=3)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 1.803, places=3)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, -153.435, places=3)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(-1.0, -0.5, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(-1.0, 0.5, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.collision_info.time, 5.31)
        self.assertAlmostEqual(self.sim.ball1.angle, 90.0)
        self.assertAlmostEqual(self.sim.ball2.angle, -153.435, places=3)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(-1.0, -0.5, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball1.position - vp.vector(-2.31, 1.345, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.collision_info.ball2.position - vp.vector(-3.3, 1.31, 0.0)).mag, 0.0, places=3)
        self.assertAlmostEqual((self.sim.collision_info.ball1.velocity - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.collision_info.ball2.velocity - vp.vector(-1.0, -0.5, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.intersection_info, None)

    def test17_easy_intersection(self):
        self.sim = BallCollisionSimulator.create_simulator(
            PhysicsParameters(1.0, (3.0, 4.0),   (-1.0, -1.0)),  # Ball 1: mass, position, velocity
            PhysicsParameters(1.0, (-3.3, -4.0),  (0.0, 1.0)),  # Ball 2: mass, position, velocity
            10.0                               # Simulation time
        )
        self.sim.run()

        self.assertAlmostEqual(self.sim.initial_distance, 10.183, places=3)
        self.assertAlmostEqual(self.sim.ball1.radius + self.sim.ball2.radius, 1.0)
        self.assertAlmostEqual(self.sim.relative_speed, 2.236, places=3)
        self.assertLess(self.sim.dot_product, 0.0)
        self.assertAlmostEqual(self.sim.ball1_state_t0.angle, -135.0)
        self.assertAlmostEqual(self.sim.ball2_state_t0.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1_state_t0.momentum - vp.vector(-1.0, -1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2_state_t0.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.total_momentum - vp.vector(-1.0, 0.0, 0.0)).mag, 0.0)
        self.assertEqual(self.sim.collision_info, None)
        self.assertAlmostEqual(self.sim.ball1.angle, -135.0)
        self.assertAlmostEqual(self.sim.ball2.angle, 90.0)
        self.assertAlmostEqual((self.sim.ball1.momentum - vp.vector(-1.0, -1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.ball2.momentum - vp.vector(0.0, 1.0, 0.0)).mag, 0.0)
        self.assertAlmostEqual((self.sim.intersection_info.position - vp.vector(-3.3, -2.3, 0.0)).mag, 0.0)
        self.assertAlmostEqual(self.sim.intersection_info.ball1_time, 6.3)
        self.assertAlmostEqual(self.sim.intersection_info.ball2_time, 1.7)


def suite():
    my_suite = unittest.TestSuite()
    my_suite.addTest(TestBallCollisionSimulator('test01_x_axis_only'))
    my_suite.addTest(TestBallCollisionSimulator('test02_converging_lines'))
    my_suite.addTest(TestBallCollisionSimulator('test03_diverging_lines'))
    my_suite.addTest(TestBallCollisionSimulator('test04_from_sw_and_ne'))
    my_suite.addTest(TestBallCollisionSimulator('test05_right_triangle_no_collision'))
    my_suite.addTest(TestBallCollisionSimulator('test06_right_triangle_collision'))
    my_suite.addTest(TestBallCollisionSimulator('test07_small_angle_converging'))
    my_suite.addTest(TestBallCollisionSimulator('test08_parallel_lines_same_direction'))
    my_suite.addTest(TestBallCollisionSimulator('test09_parallel_lines_diff_direction'))
    my_suite.addTest(TestBallCollisionSimulator('test10_parallel_lines_same_direction_one_faster'))
    my_suite.addTest(TestBallCollisionSimulator('test11_overtaking_parallel_lines'))
    my_suite.addTest(TestBallCollisionSimulator('test12_different_masses_colliding'))
    my_suite.addTest(TestBallCollisionSimulator('test13_one_object_not_moving'))
    my_suite.addTest(TestBallCollisionSimulator('test14_both_objects_not_moving'))
    my_suite.addTest(TestBallCollisionSimulator('test15_same_initial_position'))
    my_suite.addTest(TestBallCollisionSimulator('test16_a_glancing_blow'))
    my_suite.addTest(TestBallCollisionSimulator('test17_easy_intersection'))
    return my_suite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ball Collision Simulator')
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()

    if args.no_gui:
        BallCollisionSimulator.disable_gui(True)

    # Create runner and run test suite
    runner = unittest.TextTestRunner(verbosity=2, failfast=True, durations=0)
    result = runner.run (suite())

    # Quit Simulation
    BallCollisionSimulator.quit_simulation()

    # exit with "not" return code from test
    sys.exit(not result.wasSuccessful())
