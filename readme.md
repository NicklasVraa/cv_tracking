# Computer Vision Spring 2023
Authors: Nicklas Vraa, Mathias Vraa

This repository accompanies our final report in the course "Computer Vision" at Aarhus University, where we attempt to explore and understand state-of-the-art algorithms for multi-object tracking.

This repository enables the reader to reproduce the results that are described in our report, as well as trying a live demo of each of the tracking algorithms.

## Environment Setup (Linux, recommended)
Reproducing the environment in which we did our tests is paramount for the code to be able to run. It was tested on Debian-based Linux using python 3.10. We have included a list of [required](requirements.txt) packages, which can be installed easily using pip.

1. Install python 3.10 and virtual environment support, using:
    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.10 python3.10-venv
    ```

2. Create and activate a new virtual environment, then update pip:
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    ```

3. Clone this repository and install its dependencies:
    ```bash
    git clone
    cd cv_tracking
    pip install -r requirements.txt
    ```

## Environment Setup (Windows)
Not recommended, but possible. Installation and setup of CUDA has to be done manually and prior on windows. This was tested on Windows 10.

1. Install [Python 3.10](https://www.python.org/downloads/release/python-31011/) manually, then check if you are running the correct version:
    ```bash
    python --version
    ```

2. Create and activate a new virtual environment, then update pip:
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    python -m venv venv
    venv\Scripts\activate
    python.exe -m pip install --upgrade pip
    ```

3. Install git and clone this repository and install its dependencies:
    ```bash
    git clone
    cd cv_tracking
    pip install -r requirements-win10.txt
    ```

    Run `python track.py`, and if any modules from [requirements.txt](requirements.txt) have failed to install, attempt to install them manually with `pip install module_name==version`.

## Use
This section details how to use this repository, when inside the dedicated virtual environment (linux):

1. Track objects in a video input using a given tracking stack, i.e. a demo:
    ```bash
    python track.py <stack-args> <additional-args>
    ```
    Stack arguments: These arguments sets up a tracking stack, which consists of a detector and a tracking algorithm.
    - `--detector`: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
    - `--tracker`: `strongsort`, `botsort`, `deepocsort`

    Additional arguments:
    - `--source`: `0` for webcam, `<link>` for video stream, `<video>` for video file, `<path>` for a series of images.

2. Evaluate a given tracking stack on a given dataset:
    ```bash
    python evaluate.py <stack-arguments> <additional-args>
    ```
    Additional arguments:
    - `--benchmark`: `MOT16`, `MOT17`, `MOT20`, `MOT17-mini`
    - `--threads`: `1`, `2`, ...

### Reproducing our exact measurements
Run the evaluation script using `--detector yolov8x.pt` and one of the trackers on a particular benchmark. Depending on your GPU, it may take several hours to complete all evaluations. Remember to activate your environment and have the working directory be the root of this repository. If your hardware supports it, multiple threads can be used, which will speed up evaluation, but also print outputs multiple times. This is done by adding the `--threads` argument.

## Demo
| StrongSORT | BoT-SORT | Deep OC-SORT |
|------------|----------|--------------|
| ![1](resources/strongsort_demo.gif) | ![2](resources/botsort_demo.gif) | ![3](resources/deepocsort_demo.gif) |

See the full-sized demos in resources folder, or try it on your own video by running:
```bash
python track.py <stack-args> --source path/to/video --save_conf --show
```
