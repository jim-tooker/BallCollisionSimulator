#!/bin/bash
cd ..
pdoc -f --html -o docs ball_collision_sim.py
pdoc -f --html -o docs test_ball_collision_sim.py
pdoc -f --html -o docs test_cases.py
