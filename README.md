# mce

A Minimal CMIP Emulator (MCE) is a simplified climate model that mimics an ensemble of state-of-the-art full-scale complex climate models from the Coupled Model Intercomparison Project (CMIP). It is intended to emulate the time series of several key variables, such as effective radiative forcing and global-mean temperature response, from individual CMIP models in a minimal manner with sufficient accuracy. The main uses include diagnosing forcing and response parameters of CMIP models and conducting probabilistic climate projections that reflect multi-model variations.

The MCE has evolved as a climate scenario analysis tool, expanding its functionality and range of applications. The latest v1.3 allows for a more comprehensive analysis of causal relationships from various emissions to global climate and carbon cycle responses. It also incorporates a flexible mechanism for handling various scenario data and parameter ensembles for probabilistic assessment.

For more information on how to use this tool, refer to the following Jupyter Notebook files and the documentation of Python modules imported therein:

- [`calib_climate_pre.ipynb`](notebook/calib_climate_pre.ipynb): normalizes CMIP5 and CMIP6 climate model outputs
- [`calib_climate.ipynb`](notebook/calib_climate.ipynb): calibrates impulse response parameters for CMIP5 and CMIP6 climate models
- [`mk_forcing_ar6__01.ipynb`](notebook/mk_forcing_ar6__01.ipynb): prepares the data needed to calculate categorized forcing based on Indicators of Global Climate Change
- [`mk_forcing_ar6__02.ipynb`](notebook/mk_forcing_ar6__02.ipynb): creates historical scenario data based on Indicators of Global Climate Change
- [`mk_inv_emissions.ipynb`](notebook/mk_inv_emissions.ipynb): creates historical emissions of non-CO<sub>2</sub> greenhouse gases by inverting their observed concentrations, including CH<sub>4</sub> and N<sub>2</sub>O emissions from natural sources and emissions of 49 halogenated species
- [`mk_scenario_ar6db.ipynb`](notebook/mk_scenario_ar6db.ipynb): prepares illustrative scenarios from IPCC WGIII AR6
- [`mk_scenario_rcmip2.ipynb`](notebook/mk_scenario_rcmip2.ipynb): prepares scenarios from RCMIP Phase 2
- [`mk_scenario_hist_future.ipynb`](notebook/mk_scenario_hist_future.ipynb): combines historical and future scenarios over a transition period
- [`t_forcing.ipynb`](notebook/t_forcing.ipynb): a use case describing the CO<sub>2</sub> forcing scheme in MCE
- [`t_climate.ipynb`](notebook/t_climate.ipynb): a use case for describing the climate component in MCE
- [`pulse_response.ipynb`](notebook/pulse_response.ipynb): a use case for calculating thermal responses to idealized forcing changes
- [`t_driver_gascycle.ipynb`](notebook/t_driver_gascycle.ipynb): a use case describing the gas-cycle component in MCE for non-CO<sub>2</sub> greenhouse gases
- [`t_driver_climate.ipynb`](notebook/t_driver_climate.ipynb): a use case for CO<sub>2</sub>-only climate runs
- [`t_driver.ipynb`](notebook/t_driver.ipynb): a use case for CO<sub>2</sub>-only climate-carbon cycle runs in both emission- and concentration-driven modes
- [`t_driver_full_emissions.ipynb`](notebook/t_driver_full_emissions.ipynb): a use case for an emission-driven climate-carbon cycle run using full greenhouse gases and other anthropogenic and natural forcing agents
- [`t_genparms.ipynb`](notebook/t_genparms.ipynb): a use case for parameter sampling and constraining based on CMIP5 and CMIP6 Earth system models

The following Notebook files were used to create figures in published papers (Tsutsui, 2020; 2022):

- [`mkfig.ipynb`](notebook/mkfig.ipynb): figures in Tsutsui (2020), including updates
- [`mkfig_pdf.ipynb`](notebook/mkfig_pdf.ipynb): supplementary figures in Tsutsui (2020), including updates
- [`t_genparms_rcmip2.ipynb`](notebook/t_genparms_rcmip2.ipynb): figures in Tsutsui (2022), as well as descriptions of the main MCE components and the results from ensemble runs constrained according to the RCMIP2 protocol


Tsutsui, J. (2020). Diagnosing transient response to CO<sub>2</sub> forcing in coupled atmosphere-ocean model experiments using a climate model emulator. Geophys. Res. Lett., 47, e2019GL085844. https://doi.org/10.1029/2019GL085844

Tsutsui, J. (2022). Minimal CMIP Emulator (MCE v1.2): a new simplified method for probabilistic climate projections, Geosci. Model Dev., 15, 951-970. https://doi.org/10.5194/gmd-15-951-2022


## First release v1.0

The first release v1.0 contains Python programs and associated data with the emulator as well as Jupyter notebooks to demonstrate typical usage. The programs consist of core modules and specific scripts for several tasks. The core modules provide computing effective radiative forcing of atmospheric CO<sub>2</sub> and temperature response to changes in the CO<sub>2</sub> concentration. The specific scripts perform tasks for preprocessing CMIP models' output and diagnosing their forcing-response parameters. The preprocessing has two steps: global averaging and anomaly computing. The associated data include a set of diagnosed parameters for currently 25 models from CMIP phase 5 (CMIP5) and 22 models from phase 6 (CMIP6). Preprocessed CMIP data are provided from different repositories depending on the license of each model output.

Computing methods and analysis of the CMIP5 and CMIP6 models are described in the following paper:


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

