<div align="center">

# Safe and Scalable Multi-Drone Payload Transport

[![Paper](https://img.shields.io/badge/IEEE%20RA--L-2026-blue)](https://ieeexplore.ieee.org/abstract/document/11614755)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLRA.2026.3715346-informational)](https://doi.org/10.1109/LRA.2026.3715346)
[![Built on DGPPO](https://img.shields.io/badge/built%20on-DGPPO-lightgrey)](https://github.com/MIT-REALM/dgppo)

Code release for:

Jaeyoun Choi, Oswin So, Songyuan Zhang, Clark Taylor and Chuchu Fan,
**"Safe and Scalable Multi-Drone Payload Transport via CBF-Based Reinforcement
Learning With Zero-Shot Sim-to-Real Transfer,"**
*IEEE Robotics and Automation Letters*, vol. 11, no. 9, pp. 10831-10838, Sept. 2026.

[Paper](https://ieeexplore.ieee.org/abstract/document/11614755) •
[Citation](#citation) •
[Installation](#installation-quick-start) •
[Reproducing the paper](#reproducing-the-paper-results) •
[What is new vs DGPPO](#what-is-new-relative-to-upstream-dgppo)

</div>

## Overview

A team of drones carries a shared payload on compliant cables and must deliver
it to a goal pose while avoiding obstacles sensed onboard by LiDAR. The policy
is trained with a CBF-based safe reinforcement learning method and transfers to
hardware **zero-shot**, with no real-world fine-tuning.

Two design choices in the environment carry most of that result, and both are
exposed as command-line flags:

- **Cable compliance is randomized.** The spring stiffness coupling each drone
  to the payload is resampled every episode from
  `[--min-stiffness, --max-stiffness]`, so the policy never overfits to one
  cable model. This is the domain randomization behind the zero-shot transfer.
- **Team size is randomized.** The number of active drones is resampled every
  episode from `[--min-num-agents, --max-num-agents]`. One policy therefore
  serves any team size in that range, which is what makes it scalable; at
  evaluation time it also generalizes to team sizes never seen in training.

The simulation is planar: each drone is a double integrator commanded in
acceleration, the payload is a rigid body with position, velocity, orientation
and angular velocity, and each drone observes the world through a 32-beam LiDAR
of which the nearest 8 returns enter the graph.

The learning backbone is **DGPPO** ([MIT-REALM/dgppo](https://github.com/MIT-REALM/dgppo),
ICLR 2025). This repository is derived from it: the upstream README is retained
in full below, and all DGPPO algorithm code remains upstream's under upstream's
LICENSE. Our contribution is the environment, its physics, and the training and
evaluation setup documented here.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{choi2026safe,
  author  = {Choi, Jaeyoun and So, Oswin and Zhang, Songyuan and Taylor, Clark and Fan, Chuchu},
  title   = {Safe and Scalable Multi-Drone Payload Transport via {CBF}-Based Reinforcement Learning With Zero-Shot Sim-to-Real Transfer},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {11},
  number  = {9},
  pages   = {10831--10838},
  year    = {2026},
  doi     = {10.1109/LRA.2026.3715346}
}
```

Please also cite the DGPPO backbone this work builds on:

```bibtex
@inproceedings{zhang2025dgppo,
  author    = {Zhang, Songyuan and So, Oswin and Black, Mitchell and Fan, Chuchu},
  title     = {Discrete {GCBF} Proximal Policy Optimization for Multi-agent Safe Optimal Control},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

## Installation (quick start)

Tested with Python 3.10 and JAX 0.6.2, on CPU and on CUDA GPUs.

```bash
conda create -n dgppo python=3.10 && conda activate dgppo
git clone https://github.com/JY-HIM4U/DGPPO_MRTransport.git
cd DGPPO_MRTransport
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins `jax[cuda]`. For a CPU-only machine install plain
`jax` instead, or prefix any command below with `JAX_PLATFORMS=cpu`.

Check the install by replaying the released policy (see below) — it needs no
training and finishes in under a minute.

## Reproducing the paper results

### Evaluate the released policy

The trained policy behind the reported results is checked in, so the numbers
can be reproduced without retraining:

```bash
python test.py --path pretrained/VMASCollaborativeTransportLidar_dgppo_seed0 --epi 5 -n 5
```

At the default seed (1234) this prints `reward -7.780 / cost -0.508` for
episode 0 and `reward -5.675 / cost -0.549` for episode 1, both at a 100% safe
rate. Deviations of order 0.01 in reward are ordinary CPU/GPU floating-point
differences; cost and safe rate should match exactly.

Because team size is resampled per episode, the `n_real=...` field in the
output is the number of drones actually active, which will vary. Passing a
different `-n` than the policy was trained with is how the scalability results
were produced.

Add `--no-video` to skip rendering, or drop it to write an MP4 per episode
under `<path>/videos/`.

### Train from scratch

```bash
python train.py \
  --env VMASCollaborativeTransportLidar --algo dgppo \
  -n 5 --obs 5 --n-rays 32 \
  --min-num-agents 3 --max-num-agents 5 \
  --min-stiffness 0.05 --max-stiffness 0.3 \
  --agent-vertex-constraint 0.2 \
  --steps 200000 --n-env-train 128 --n-env-test 32 \
  --batch-size 16384 --rnn-step 16 \
  --eval-interval 50 --save-interval 50 --seed 0
```

Note that `--max-stiffness 0.3` and `--agent-vertex-constraint 0.2` differ from
the defaults in `make_env`, so pass them explicitly. Every run writes its fully
resolved configuration to `<log_dir>/.../config.yaml`; the config for the
released policy is in `pretrained/VMASCollaborativeTransportLidar_dgppo_seed0/`.

Runs are logged to Weights & Biases when a connection is available and fall
back to offline mode otherwise. Set `WANDB_MODE=disabled` to turn it off.

## What is new relative to upstream DGPPO

| Path | Role |
| --- | --- |
| `dgppo/env/vmas_lidar/vmas_collaborative_transport_lidar.py` | The environment: payload dynamics, cable-spring coupling, LiDAR observation, reward and cost terms |
| `dgppo/env/vmas_lidar/physax/` | Forked 2-D rigid-body physics and raycasting (`world.py`, `entity.py`, `geometry.py`, `shapes.py`, `raycast.py`) |
| `dgppo/env/obstacle.py` | Adds `Circle` and `Polygon` obstacles |
| `dgppo/env/utils.py` | `get_lidar` extended to circles and polygons; `get_node_goal_rng` reworked for boundary margins and minimum agent-goal travel |
| `dgppo/env/__init__.py` | Environment registration plus the reward, stiffness and vertex-constraint knobs on `make_env` |
| `train.py`, `test.py` | Command-line flags for the above, plus evaluation and CSV/plot export |
| `pretrained/` | The policy behind the reported results, with its resolved config |

## Relationship to DGPPO

This repository is a derived copy of
[MIT-REALM/dgppo](https://github.com/MIT-REALM/dgppo), the official JAX
implementation of *Discrete GCBF Proximal Policy Optimization for Multi-agent
Safe Optimal Control* (ICLR 2025), which provides the learning backbone. It is
not a GitHub fork for an administrative reason only: that account already holds
a fork of the upstream repository.

All DGPPO algorithm code here is upstream's, under upstream's terms; see
[LICENSE](LICENSE). Our contribution is the payload-transport environment, its
physics, and the training and evaluation setup documented above.

The upstream environments (`MPE*`, `Lidar*`, `VMASWheel`,
`VMASReverseTransport`) and algorithms (`informarl`, `informarl_lagr`,
`hcbfcrpo`, `dgppo`) are all still present and usable here. They are documented
in the [upstream README](https://github.com/MIT-REALM/dgppo/blob/main/README.md),
which we link to rather than copy so it cannot drift out of date.

## License

See [LICENSE](LICENSE), inherited from the upstream DGPPO project.
