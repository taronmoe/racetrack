# Reinforcement Learning for the Racetrack Problem

This project implements and evaluates three reinforcement learning algorithms
for the racetrack problem: Value Iteration, Q-Learning, and SARSA.

The environment models a car with position and velocity states, stochastic
acceleration failures, and configurable crash penalties, resulting in a large
and highly structured state space.

## Algorithms Implemented
- **Value Iteration**  
  Exact, model-based planning using Bellman backups over the full state space.

- **Q-Learning**  
  Off-policy, model-free learning using temporal-difference updates.

- **SARSA**  
  On-policy, model-free learning emphasizing policy stability.

## Environment & Dynamics
- State: (x position, y position, x velocity, y velocity)
- Actions: discrete accelerations
- Transition noise: probabilistic acceleration failure
- Crash handling: restart at start line or nearest valid cell

## Evaluation
The algorithms were evaluated across multiple track layouts (2-track, W-track,
U-track) and crash penalty models. Results highlight clear tradeoffs between
policy optimality, convergence speed, and computational cost.

Value Iteration produces optimal policies but scales poorly. Q-Learning and
SARSA scale better to large tracks, with SARSA producing more stable but
conservative policies.

## Key Takeaway
Exact planning is ideal when the state space is manageable. Model-free
reinforcement learning offers a scalable alternative when dynamics are complex
and the state space grows large.

## Notes
This project emphasizes explicit state modeling, correctness, and reproducible
experimentation, and was implemented from scratch in Python as part of an
Artificial Intelligence course.
