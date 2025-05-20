# mce

## Overview

**Minimal CMIP Emulator (MCE)** is a simplified climate modeling tool designed to emulate the behavior of complex Earth system models from the Coupled Model Intercomparison Project (CMIP). It reproduces the time series of key climate variables—such as effective radiative forcing and global-mean temperature response—from individual CMIP models in a minimal manner with sufficient accuracy.

MCE is primarily used to:
- Diagnose forcing and response parameters of CMIP-class models
- Conduct probabilistic climate projections that reflect inter-model variability

The model's components and typical applications are described in Tsutsui (2020; 2022).

Over time, MCE has evolved into a versatile tool for climate scenario analysis. The latest version (v1.3) introduces enhanced capabilities for tracing causal relationships between emissions and global climate and carbon cycle responses. To support this, the **Driver Class** responsible for time integration has been fundamentally redesigned, enabling a wide range of simulations—from simple baseline runs to complex configurations with multiple conditional settings. This redesign offers greater flexibility and extensibility. The latest updates also introduce a robust mechanism for handling diverse scenario data and parameter ensembles, thereby facilitating probabilistic assessments.

---

## Jupyter Notebook guide

For detailed guidance on how to use this tool, refer to the categorized Jupyter Notebook files below, along with the documentation of the Python modules they rely on.

### Scenario data processing

| Notebook | Description |
|----------|-------------|
| [`mk_forcing_ar6__01.ipynb`](notebook/mk_forcing_ar6__01.ipynb) | Preparation of data needed to calculate categorized forcing based on Indicators of Global Climate Change (Forster et al., 2024) |
| [`mk_forcing_ar6__02.ipynb`](notebook/mk_forcing_ar6__02.ipynb) | Construction of historical scenario data based on Indicators of Global Climate Change |
| [`mk_inv_emissions.ipynb`](notebook/mk_inv_emissions.ipynb) | Inversion-based construction of historical non-CO<sub>2</sub> greenhouse gas emissions |
| [`mk_scenario_ar6db.ipynb`](notebook/mk_scenario_ar6db.ipynb) | Preparation of illustrative scenarios from IPCC WGIII scenario database (Byers et al., 2022) |
| [`mk_scenario_rcmip2.ipynb`](notebook/mk_scenario_rcmip2.ipynb) | Preparation of scenarios from RCMIP Phase 2 (Nicholls et al., 2021) |
| [`mk_scenario_hist_future.ipynb`](notebook/mk_scenario_hist_future.ipynb) | Combining historical and future scenarios across a transition period |

### Model overview and fundamentals

| Notebook | Description |
|----------|-------------|
| [`t_forcing.ipynb`](notebook/t_forcing.ipynb) | CO<sub>2</sub> forcing scheme in MCE |
| [`t_climate.ipynb`](notebook/t_climate.ipynb) | Climate component in MCE |
| [`pulse_response.ipynb`](notebook/pulse_response.ipynb) | Comparison of thermal responses to pulse-like CO<sub>2</sub> forcing between three- and two-layer climate models |

### Driver usage

| Notebook | Description |
|----------|-------------|
| [`t_driver.ipynb`](notebook/t_driver.ipynb) | Simulating climate and carbon-cycle responses to CO<sub>2</sub> forcing using their coupled components |
| [`t_driver_climate.ipynb`](notebook/t_driver_climate.ipynb) | Simulating climate response to CO<sub>2</sub> forcing using the climate component only |
| [`t_driver_gascycle.ipynb`](notebook/t_driver_gascycle.ipynb) | Simulating gas-cycle response to non-CO<sub>2</sub> greenhouse gas emissions |
| [`t_driver_full_emissions.ipynb`](notebook/t_driver_full_emissions.ipynb) | Simulating climate, carbon-cycle, and gas-cycle responses to all forcing agents using their fully coupled components |

### Parameter ensemble

| Notebook | Description |
|----------|-------------|
| [`calib_climate_pre.ipynb`](notebook/calib_climate_pre.ipynb) | Pre-processing CMIP model outputs for calibration |
| [`calib_climate.ipynb`](notebook/calib_climate.ipynb) | Calibration of impulse response parameters to CMIP climate models |
| [`t_genparms.ipynb`](notebook/t_genparms.ipynb) | Generating a large parameter ensemble based on CMIP Earth system models |


### Reproduction of published results

| Notebook | Description |
|----------|-------------|
| [`mkfig.ipynb`](notebook/mkfig.ipynb) | Generating figures used in Tsutsui (2020) including updates |
| [`mkfig_pdf.ipynb`](notebook/mkfig_pdf.ipynb) | Generating supplementary figures in Tsutsui (2020) including updates |
| [`t_genparms_rcmip2.ipynb`](notebook/t_genparms_rcmip2.ipynb) | Generating a large parameter ensemble used in RCMIP Phase 2 |
| [`ensemble_runs.ipynb`](notebook/ensemble_runs.ipynb) | Ensemble runs applied to RCMIP Phase 2, conducted with an old driver class |


### Note on dependencies

- **External or unpublished data**: Some notebooks in this repository rely on data that are not fully included in the current release due to licensing, size, or publication constraints.
- **Legacy dependencies**: Some notebooks were developed using earlier versions of the Driver module. While many have been updated to work with the latest modules and data structures, some still depend on legacy implementations.
- **Version differences**: Due to differences in Python library versions or changes in input/output data formats, numerical results may not exactly match those obtained with previous releases, even when using the same model configuration.

---

## References

1. Byers, E., Krey, V., Kriegler, E., Riahi, K., Schaeffer, R., Kikstra, J., et al. (2022). AR6 Scenarios Database hosted by IIASA. International Institute for Applied Systems Analysis. https://doi.org/10.5281/zenodo.5886912
2. Forster, P. M., Smith, C., Walsh, T., Lamb, W. F., Lamboll, R., Hall, B., et al. (2024). Indicators of Global Climate Change 2023: annual update of key indicators of the state of the climate system and human influence. *Earth Syst. Sci. Data*, 16, 2625–2658. https://doi.org/10.5194/essd-16-2625-2024
3. Nicholls, J. et al. (2021) Reduced Complexity Model Intercomparison Project Phase 2: Synthesizing Earth system knowledge for probabilistic climate projections. *Earth’s Future* 9, e2020EF001900. https://doi.org/10.1029/2020EF001900
4. Tsutsui, J. (2020). Diagnosing transient response to CO<sub>2</sub> forcing in coupled atmosphere-ocean model experiments using a climate model emulator. *Geophys. Res. Lett.*, 47, e2019GL085844. https://doi.org/10.1029/2019GL085844
5. Tsutsui, J. (2022). Minimal CMIP Emulator (MCE v1.2): a new simplified method for probabilistic climate projections, *Geosci. Model Dev.*, 15, 951–970. https://doi.org/10.5194/gmd-15-951-2022

*Additional references may be included in individual notebooks.*

---

## Disclaimer

This software and its associated Jupyter Notebooks are intended for **research and educational purposes only**. While efforts have been made to ensure scientific accuracy and reproducibility, the model and its outputs are simplified representations of complex climate systems and should **not be used for operational forecasting, policy decision-making, or commercial applications** without further validation.

Use of this tool is at your own risk. The author assumes no liability for any direct or indirect consequences arising from its use.

---

## Release notes

### Latest release (v1.3)

The following is a summary of the main updates to the Python core modules included in this release:

- **Updated**: `__init__.py` - Added classes for robust handling of model parameters and scenario I/O
- **Added**: `calib.py` - Introduced a base class for parameter calibration
- **Updated**: `carbon.py` - Integrated new parameter handling; added a method for ocean parameter calibration
- **Updated**: `climate.py` - Integrated new parameter handling; added a method to derive impulse response parameters from box model parameters; Added a parameter for ocean heat uptake efficacy
- **Rewritten**: `driver.py` - Replaced the previous driver class with a fully redesigned, inheritance-based architecture for flexible and modular simulation workflows
- **Updated**: `forcing.py` - Integrated new parameter handling; added schemes for non-CO<sub>2</sub> greenhouse gases; added CO<sub>2</sub> scheme used in IPCC Sixth Assessment Report (AR6)
- **Added**: `forcing_ar6.py` - Introduced a subclass to evaluate AR6-based categorized forcing

Most of the Jupyter Notebooks listed above have been added or revised to reflect updates to the core modules.  
The utility modules have also been updated, including:

- Enhanced handling of data input and output  
- Expanded plotting and visualization functions  
- Removal of redundant or outdated scripts and code segments  

These changes aim to improve usability, maintainability, and consistency across the modeling workflow.


### Previous releases

- **Minor update v1.2.1** (Oct 18, 2021)  
  Added [`t_genparms_rcmip2.ipynb`](notebook/t_genparms_rcmip2.ipynb) to reproduce the parameter ensemble used in the RCMIP Phase 2.

- **Second release v1.2** (Mar 15, 2021)  
  Added carbon-cycle and driver modules for scenario runs. The driver supports multi-agent forcing and ensemble runs with perturbed model parameters for probabilistic climate projections.

  Typical usage is demonstrated in Jupyter Notebook files:
  - [`t_driver.ipynb`](notebook/t_driver.ipynb) for the driver
  - [`t_ensemble_runs.ipynb`](notebook/t_ensemble_runs.ipynb) for ensemble runs
  - [`t_genparms.ipynb`](notebook/t_genparms.ipynb) for perturbed model parameters

- **First release v1.0** (Jan 2, 2021)  
  Initial public release, including core modules, calibrated parameter data, scripts for several tasks, and Jupyter Notebooks to demonstrate typical usage.
