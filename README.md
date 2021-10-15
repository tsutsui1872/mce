# mce
A Minimal CMIP Emulator (MCE) is a simplified climate model that mimics an ensemble of state-of-the-art, full-scale complex climate models from the Coupled Model Intercomparison Project (CMIP). It is intended to emulate time series of several key variables, such as effective radiative forcing and global-mean temperature response, from individual CMIP models in a minimal way with sufficient accuracy. Main use includes diagnosing forcing and response parameters of CMIP models and conducting probabilistic climate projections that reflect multi-model variations.

## First release v1.0

The first release v1.0 contains Python programs and associated data with the emulator as well as Jupyter notebooks to demonstrate typical usage. The programs consist of core modules and specific scripts for several tasks. The core modules provide computing effective radiative forcing of atmospheric CO<sub>2</sub> and temperature response to changes in the CO<sub>2</sub> concentration. The specific scripts perform tasks for preprocessing CMIP models' output and diagnosing their forcing-response parameters. The preprocessing has two steps: global averaging and anomaly computing. The associated data include a set of diagnosed parameters for currently 25 models from CMIP phase 5 (CMIP5) and 22 models from phase 6 (CMIP6). Preprocessed CMIP data are provided from different repositories depending on the license of each model output.

Computing methods and analysis of the CMIP5 and CMIP6 models are described in the following paper:

Tsutsui, J. (2020). Diagnosing transient response to CO<sub>2</sub> forcing in coupled atmosphere-ocean model experiments using a climate model emulator. Geophysical Research Letters, 47, e2019GL085844. https://doi.org/10.1029/2019GL085844

One of the Jupyter notebooks demonstrates creating the figures shown in this paper.

Future release will include data for more CMIP models to be emulated, carbon cycle modules to compute natural CO<sub>2</sub> uptake in the ocean and terrestrial biosphere, and non-CO<sub>2</sub> forcing modules.


## Second release v1.2

The second release is mainly updated with the addition of carbon cycle and driver modules for scenario runs. The driver supports multi-agent forcing and ensemble runs with perturbed model parameters for probabilistic climate projections. Scenario runs can use input time series provided from Reduced Complexity Model Intercomparison Project ([RCMIP](https://www.rcmip.org/)).

Typical usage is demonstrated in Jupyter Notebook files:
- [`t_driver.ipynb`](notebook/t_driver.ipynb) for the driver
- [`t_ensemble_runs.ipynb`](notebook/t_ensemble_runs.ipynb) for ensemble runs
- [`t_genparms.ipynb`](notebook/t_genparms.ipynb) for perturbed model parameters


### Minor update v1.2.1

A new Jupyter notebook
[`t_genparms_rcmip2.ipynb`](notebook/t_genparms_rcmip2.ipynb)
has been added, which reproduces the perturbed-parameter ensemble used in the RCMIP Phase 2 (Nicholls et al. 2021, https://doi.org/10.1029/2020EF001900).
The model components and the experiment results are described in Tsutsui (2021).

Tsutsui, J.: Minimal CMIP Emulator (MCE v1.2): A new simplified method for probabilistic climate projections, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-79, in review, 2021.

