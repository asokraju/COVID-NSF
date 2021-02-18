# Instructions

## Instructions for installing ubuntu on WSL ubuntu

Partial guide from [here](https://www.how2shout.com/how-to/install-anaconda-wsl-windows-10-ubuntu-linux-app.html)

- Downloading anaconda in ubuntu
  - `wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh`
  - list the files `ls`

- Installing

- `sudo bash Anaconda3-2020.07-Linux-x86_64.sh` or
  - `sudo ./Anaconda3-2020.07-Linux-x86_64.sh`
  - Press `q` and then type `Yes` to accept the license

- `source ~/.bashrc`

- Anaconda [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

___

## Installation instructions using anaconda on windows/Ubuntu

- Use Anconda powershell in windows (with root access)

## Steps

- Create a new enviroment called `tf-gpu` with the latest python
  - `conda create -n tf-gpu python=3.8`

- Activate the enviroment
  - `conda activate tf-gpu`

- Installing cuda tool-kit
  - Installing cudnn in the next step will also install the appropriate cuda version. So skip this, unless you want a particulat version, then you also need specify the version for cudnn.
  - Download and install cuda - `11.x.x` from nvidea-website
  - (or using conda) `conda install cudatoolkit=11.0`
  <!-- - Restart the pc -->

- Installing cudnn
  - `conda install cudnn`

- Anaconda maintains a stable (currently `2.3.x`, not the latest i.e., `2.4.x`) version of tensorflow with/without gpu and cudnn binaries
  - (just cpu) `conda install tensorflow`
  - (or: to use gpu) `conda install tensorflow-gpu`

- The enviroment was written using Open-AI gym (`v0.18`). Use one of the following
  - `conda install gym` (Works in windows)
  - (or) `conda install -c conda-forge gym`
  - (or) `pip install gym` (works in ubuntu)

- Install joblib (`1.0.0`) to save/load data
  - `conda install joblib`

- Data-manipulation, plotting -- matplotlib (`3.3.2`), pandas (`1.2.1`)
  - `conda install matplotlib`
  - `conda install pandas`

- We use sklearn (`0.23.2`) to standardize the data
  - `conda install scikit-learn`

- Use ipython (`7.20.0`) kernel for debugging and testing
  - `conda install ipython`

## Verification

1. Using `numba`
    - `numba -s`

    you should something similar:

    ```text
    __CUDA Information__
    CUDA Device Initialized                       : True
    CUDA Driver Version                           : 11020
    CUDA Detect Output:
    Found 1 CUDA devices
    id 0     b'GeForce GTX 980M'                              [SUPPORTED]
    ```

2. Using `python`

    ```python
    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    ```

## Final list of necessary packages

```text
gym                       0.18.0
ipython                   7.20.0
joblib                    1.0.0
matplotlib                3.3.2
numpy                     1.19.2
pandas                    1.2.1
python                    3.8.5
scikit-learn              0.23.2
scipy                     1.6.0
setuptools                52.0.0
tensorflow                2.3.0
tensorflow-gpu            2.3.0
cudnn                     8.0.5.39
cudatoolkit               11.0.221
```

___

## Installing git and cloning the files in ubuntu

Mostlikely `git` is available in ubuntu. If not,

- `sudo apt install git`
- removing existing files `rm -rf COVID-AWS`
- `git clone https://github.com/asokraju/COVID-AWS.git`

___

## Running the code in the batch mode in Ubuntu

- find the existing pythons installed:
  - `which python`
  - This return something like 
  */root/anaconda3/envs/tf-gpu/bin/python*. Let us call it as `PYTHON_PATH`
  - copy it using `Ctrl + Right-CLick`
  - open the file using

    ```bash
    cd COVID-AWS
    nano bash-scripts/BaselineSenario_Batch.sh
    ```

    change the last line 
  - Change *python path* in *line 43* of *bash-scripts/BaselineSenario_Batch.sh* to
    `PYTHON_PATH $run_exec $run_flags`
  - Select the text and past using *Ctrl + Shift + v*
  - `chmod u+x bash-scripts/BaselineSenario_Batch.sh`
  - `chmod u+x bash-scripts/BaselineSenario_JobScript.sh`
  - `sudo bash bash-scripts/BaselineSenario_JobScript.sh`
