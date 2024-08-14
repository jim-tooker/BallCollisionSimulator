#!/bin/bash
# Note: This script is meant to be run from the root project directory, like this `tools/generate_docs.sh`.
pdoc -f --html -o docs ball_collision_sim.py
pdoc -f --html -o docs test_ball_collision_sim.py
pdoc -f --html -o docs test_cases.py
