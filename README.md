# PINN_3BodyProblem

Welcome to my repository! In this project, I used Physics-Informed Neural Networks (PINNs) to learn the dynamics of the Three-Body Problem.

### `Simple3BodyProblem`:
This folder contains simulations of the 3-body problem in 2D space with the following initial conditions:
- Initial velocity: 0
- Masses: All set to 1
- PINN model predicts the system's behavior up to 10 seconds.

### `General3BodyProblem`:
In this folder, things get more spicier! The problem is now simulated in 3D, with the PINN predicting the system's dynamics up to 50 seconds. This version introduces variability in the starting velocities and masses of the bodies, making it a more general and challenging setup.

### Paper:
The work and experiments are detailed in the paper titled `PINN_for_General_3Body_Dynamics_Paper.pdf`, which highlights the methodology and results of this project.
