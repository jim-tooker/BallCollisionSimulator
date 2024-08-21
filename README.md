
# Ball Collision Simulator
(Package `ballcollide`)

This package simulates the elastic, inelastic, or partially elastic collisions between two balls using the VPython library. It contains modules to define the physical and visual properties of the balls, perform the simulation, and visualize the collision in a 2D space.

- ## Modules:
    * `ball_collision_sim`: The main module that runs the system. It manages the entire simulation process, including initialization, running the simulation, and handling collisions.
    * `ball_sim_enums`: Contains enums used by the system.
    * `ball_sim_parameters`: Contains classes to store parameter data needed by the system.
    * `ball_sim`: Contains the `Ball` class which represents the balls in the system.
    * `ball_sim_info`: Contains classes to hold info/state of the system.
    * `tests`: Tests for the Ball Collision Simulator system.

## Documentation
For detailed API documentation, see:
[Ball Collision Simulator API Documentation](https://jim-tooker.github.io/ballcollide/docs/ballcollide/index.html)

## Sample Screenshot
<img width="593" alt="Screenshot 2024-08-21 at 8 59 51 AM" src="https://github.com/user-attachments/assets/a516d23f-0fe0-47d2-af4c-532d17358fd7">

```
***************************************************
Initial Conditions:

Ball 1:
  Mass: 1 kg
  Radius: 0.5 m
  Position: (0.4, 3)
  Velocity: (0, -2), or 2 m/s at -90°
  Momentum: (0, -2), or 2 N⋅s at -90°
  Kinetic Energy: 2 J
Ball 2:
  Mass: 2 kg
  Radius: 0.63 m
  Position: (-0.5, -3)
  Velocity: (0, 1), or 1 m/s at 90°
  Momentum: (0, 2), or 2 N⋅s at 90°
  Kinetic Energy: 1 J

Initial Distance: 6.07 m
Sum of Radii: 1.13
Total Momentum: (0, 0), or 0 N⋅s at 0°
Total Kinetic Energy: 3 J

t=0s: Balls are Converging at: 3 m/s
t=1.78s: Balls are Diverging at: 3 m/s

Collision occurred at time: 1.78 secs

Post Collision Conditions:

Ball 1:
  Mass: 1 kg
  Radius: 0.5 m
  Position: (0.4, -0.56)
  Velocity: (1.91, -0.601), or 2 m/s at -17.5°
  Momentum: (1.91, -0.601), or 2 N⋅s at -17.5°
  Kinetic Energy: 2 J
Ball 2:
  Mass: 2 kg
  Radius: 0.63 m
  Position: (-0.5, -1.22)
  Velocity: (-0.954, 0.301), or 1 m/s at 163°
  Momentum: (-1.91, 0.601), or 2 N⋅s at 163°
  Kinetic Energy: 1 J

###################################################
```
