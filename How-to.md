# Installation instructions using anaconda on windows

- Use Anconda powershell in windows (with root access)
- Download and install cuda - `11.x.x` from nvidea-website
- Restart the pc

## Steps

- Create a new enviroment called `tf-gpu` with the latest python
  - `conda create -n tf-gpu python=3.8`

- Activate the enviroment
  - `conda activate tf-gpu`

- Anaconda maintains a stable (currently `2.3.x`, not the latest i.e., `2.4.x`) version of tensorflow with/without gpu and cudnn binaries
  - (just cpu) `conda install tensorflow`
  - (or: to use gpu) `conda install tensorflow-gpu`

- The enviroment was written using Open-AI gym (`v0.18`). Use one of the following
  - `conda install gym`
  - (or) `conda install -c conda-forge gym`
  - (or) `pip install gym`

- Install joblib (`1.0.0`) to save/load data
  - `conda install joblib`

- Data-manipulation, plotting -- matplotlib (`3.3.2`), pandas (`1.2.1`)
  - `conda install matplotlib`
  - `conda install pandas`

- We use sklearn (`0.23.2`) to standardize the data
  - `conda install sklearn`

- Use ipython (`7.20.0`) kernel for debugging and testing
  - `conda install ipython`

# Instructions for running the code

To be updated
