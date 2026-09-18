
# SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM

[![arXiv](https://img.shields.io/badge/arXiv-2601.10814-b31b1b.svg)](https://arxiv.org/abs/2601.10814)
[![Project Page](https://img.shields.io/badge/Project_Page-SurfSLAM-blue)](https://umfieldrobotics.github.io/SurfSLAM/)
[![Project Page](https://img.shields.io/badge/DeepBlue-Dataset_and_Weights-00274C)](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411)
[![Underwater Stereo Code](https://img.shields.io/badge/github-Underwater_Stereo_Code-white.svg)](https://github.com/umfieldrobotics/SurfSLAM_Stereo)


<h4 align="center">
  <a href="https://www.obagoren.com/">Onur Bagoren</a><sup>*</sup>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://sethgi.me/">Seth Isaacson</a><sup>*</sup>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://sacchinbhg.github.io/">Sacchin Sundar</a>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://ycsun2113.github.io/">Yung-Ching Sun</a>
  <br><br>
  <a href="https://anja-sheppard.github.io/">Anja Sheppard</a>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://haoyuma2002814.github.io/">Haoyu Ma</a>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/abrar-shariff">Abrar Shariff</a>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://www.roahmlab.com/ram-personal">Ram Vasudevan</a>
  &nbsp;&nbsp;<b>&middot;</b>&nbsp;&nbsp;
  <a href="https://fieldrobotics.engin.umich.edu/team">Katherine A. Skinner</a>
</h4>

<p align="center">
  <sub><i><sup>*</sup>Equal contribution</i></sub>
</p>

<p align="center">
  <img src="docs/overview.png" alt="SurfSLAM overview" width="100%">
  <br>
  <sub><i>Background image courtesy of the National Oceanic and Atmospheric Administration Thunder Bay National Marine Sanctuary.</i></sub>
</p>

<details>
<summary><b>Abstract (Click to Expand)</b></summary>
Localization and mapping are core perceptual capabilities for underwater robots. Stereo cameras provide a low-cost means of directly estimating metric depth to support these tasks. However, despite recent advances in stereo depth estimation on land, computing depth from image pairs in underwater scenes remains challenging. In underwater environments, images are degraded by light attenuation, visual artifacts, and dynamic lighting conditions. Furthermore, real-world underwater scenes frequently lack rich texture useful for stereo depth estimation and 3D reconstruction. As a result, stereo estimation networks trained on in-air data cannot transfer directly to the underwater domain. In addition, there is a lack of real-world underwater stereo datasets for supervised training of neural networks. Poor underwater depth estimation is compounded in stereo-based Simultaneous Localization and Mapping (SLAM) algorithms, making it a fundamental challenge for underwater robot perception. To address these challenges, we propose a novel framework that enables sim-to-real training of underwater stereo disparity estimation networks using simulated data and self-supervised finetuning. We leverage our learned depth predictions to develop SurfSLAM, a novel framework for real-time underwater SLAM that fuses stereo cameras with IMU, barometric, and Doppler Velocity Log (DVL) measurements. Lastly, we collect a challenging real-world dataset of shipwreck surveys using an underwater robot. Our dataset features over 24,000 stereo pairs, along with high-quality, dense photogrammetry models and reference trajectories for evaluation. Through extensive experiments, we demonstrate the advantages of the proposed training approach on real-world data for improving stereo estimation in the underwater domain and for enabling accurate trajectory estimation and 3D reconstruction of complex shipwreck sites.
</details>

## Datasets

Data from this project is available at [DeepBlue](https://deepblue.lib.umich.edu/data/concern/data_sets/r781wh411), consisting of shipwreck surveys collected at the National Oceanic and Atmospheric Administration Thunder Bay National Marine Sanctuary. Data is stored as hdf5 archives for their compression, ability to stream, and minimal dependencies. See [docs/data_format.md](docs/data_format.md) for details on the data format, including how to convert your data to our format.


## Setup

We highly recommend using docker for this project and don't provide explicit support for non-docker setups. 

Check out the model submodules and apply our patches to them:
```bash
./scripts/setup_submodules.sh
```

The submodules point at their public upstream repositories. Our changes live in [`patches/`](patches/), one file per submodule, and `setup_submodules.sh` applies them. See **[docs/submodules.md](docs/submodules.md)** for details.

Then build the docker image. This pulls an image from docker hub that has most dependencies installed, then adds a user-specific configuration and mounts all the data paths.

```bash
cd docker/
./build_user.sh
```

**BEFORE** launch the docker container, you must tell the system where you downloaded data to:

```bash
cp cfg/dataset_paths.example.yaml cfg/dataset_paths.yaml
# edit cfg/dataset_paths.yaml to point at the DeepBlue download
```

Once that is done, you may launch the container. The first time you start the container it will build and install the [`turtlmap`](https://github.com/umfieldrobotics/TURTLMap) python bindings.

```bash
./run.sh # starts a container, or attaches to an existing container (allowing multiple terminals)
./run.sh restart # restarts the container, discarding any changes you have made to the local container
```

See **[docs/docker.md](docs/docker.md)** for more details.


## Running the Method

All commands below run inside the docker container. Note that if you are running outside of docker, you will need to manually build and install the `turtlmap` bindings. To run SLAM on a sequence:

```bash
python3 examples/run_surfslam.py cfg/tbnms/monohansett_long.yaml   # also: monohansett_boiler.yaml, monohansett_engine.yaml
```

Results (estimated trajectory, statistics, and a copy of the settings used) are written to `./outputs/<experiment_name>_<timestamp>/`. Useful flags:

- `--num_repeats N`: run N trials of each configuration (results go in `trial_*/` subdirectories).
- `--overrides <yaml>` (optionally with `--run_all_combos`): sweep parameters for ablations. See [cfg/README.md](cfg/README.md) for how the settings system works.
- `--duration <sec>`: only process the first part of the sequence.

To rebuild a dense map offline from a finished run's trajectory:

```bash
python examples/offline_mapping.py cfg/tbnms/mono_long.yaml outputs/<run_dir> --output_mesh mesh.ply
```

## Computing Metrics

By default, trajectory evaluation picks up every run of our method found in `outputs/` and compares it against the released ground-truth trajectories -- no configuration needed. To evaluate other methods side by side, add entries to the `algorithms` list in [analysis/traj_eval_cfg.yaml](analysis/traj_eval_cfg.yaml).

```bash
cd analysis/
python evaluate_trajectories.py traj_eval_cfg.yaml
```

This writes APE/RPE/completeness tables to `./eval_results/`, along with the ground-truth alignment transforms (`eval_results/alignments/`) needed for map evaluation.

For map evaluation, first run `evaluate_trajectories.py` with [analysis/traj_eval_cfg_map.yaml](analysis/traj_eval_cfg_map.yaml) to align each method's trajectory, then evaluate the reconstructions against the photogrammetry models:

```bash
python evaluate_trajectories.py traj_eval_cfg_map.yaml
python mapping/evaluate_maps.py ../eval_results/map_comparison/alignments
```

This computes accuracy, completeness, precision, and recall for every trial (ground-truth point clouds are resolved automatically from `dataset_paths.yaml`) and writes results to `eval_results/map_comparison/maps/`. Note that the release's `trajectory.tum` and `reconstruction.ply` are in different frames -- the trajectory is re-referenced so its first pose is the identity -- so `evaluate_maps.py` composes each alignment with the per-scene transform in [analysis/mapping/model_frame.yaml](analysis/mapping/model_frame.yaml) to reach the photogrammetry frame.

To turn those per-trial metrics into the paper's mapping table:

```bash
python compute_mapping_summary.py
python generate_mapping_latex_table.py
```

Runtime and pipeline statistics can be summarized with `python summarize_statistics.py --config stats_summary_cfg.yaml`.


## Acknowledgements

Code in this repo is based on the structure from [LONER](https://github.com/umautobots/LONER) and the opti-acoustic-inertial pipeline [TURTLMap](https://github.com/umfieldrobotics/TURTLMap).
