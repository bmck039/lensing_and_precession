# Distinguishing regularly precessing and lensed gravitational waveforms

## Description
Gravitational waves (GWs) from binary black hole (BBH) inspirals are affected by the black hole spins and orbital angular momentum, which, when misaligned, cause precession and nutation and introduce modulations in GW amplitudes and phases. In regular precession (without transitional precession or nutation), the total angular momentum has nearly constant direction and the orbital angular momentum precesses on a cone whose opening angle and frequency slowly increase on the radiation-reaction timescale. Regularly precessing BBH systems include those with a single spin, equal masses, or those trapped in spin-orbit resonances.

On the other hand, GWs can also be lensed by massive objects along the line of sight, resulting in amplification, potentially multiple images, and modulation of GWs. GWs are analyzed in the wave-optics regime and geometrical-optics regime depending on the mass of the lens and the wavelength. In axisymmetric lens models such as the point mass and singular isothermal sphere, the gravitational waveform can be described by the lens mass and the source position relative to the optic axis.

We investigate various parameters governing regular precession, including the precession amplitude, frequency, and the initial precessing phase, and lensing parameters, such as the lens mass and source position, to identify scenarios where the resulting waveforms may appear indistinguishable. The source’s chirp mass inversely correlates with the innermost stable circular orbit frequency cutoff and the inspiral waveform duration in the frequency band. At high chirp masses, waveforms may lack distinctive features, thus simplifying waveform matching. Through parameter tuning, a parameter space can be identified where the secular, oscillatory regularly precessing waveform aligns with the purely oscillatory lensed one. In addition, analytical approximations can predict the mismatch behavior between the lensed source and the regularly precessing template, as a function of the source’s chirp mass, which further elucidates the contribution of BBHs’ regular precession to waveform ambiguity.

Employing match-filtering analysis and various `PyCBC` packages, we quantify the mismatch and apply the Lindblom criterion to establish discernibility conditions for waveforms. Our study explores the parameter space to understand waveform distinguishability between regular precession and lensing, offering insights into the signal-to-noise requirement for GW detectors to effectively discern these waveforms.

Much of this code was taken from [here](https://github.com/fairytien/lensing_and_precession/tree/main) and expanded upon. 

## Getting started

### Installation

This project requires [`lalsuite`](https://pypi.org/project/lalsuite/) and [`PyCBC`](https://pycbc.org). Note that PyCBC sometimes fails to install via pip.

```bash
# Install the package in editable mode (recommended for development)
cd /path/to/summerResearch
python -m pip install -e .

```

if PyCBC fails to install it can be installed with:
```bash
python -m pip install PyCBC
```

This will:
- Install all required dependencies (numpy, scipy, matplotlib, lalsuite, etc.)
- Make `lensing_and_precession` and `scripts` packages importable from anywhere
- Create CLI commands: `calc-mcz-vs-td`, `plot-waveforms`, `convergence-test-delta-phi`, `test-minimization`

#### Optional dependencies

For additional features, install optional dependency groups:

```bash
# For JAX-based acceleration (jit_helper.py)
python -m pip install -e ".[jax]"

# For extra visualization tools
python -m pip install -e ".[extras]"

# Install all optional dependencies
python -m pip install -e ".[pycbc,jax,extras]"
```

### Running the analysis scripts

After installation, you can run scripts via their CLI commands:

```bash
# Show help
calc-mcz-vs-td -h
plot-waveforms -h
test-minimization -h

# Example runs
calc-mcz-vs-td -pbar -I 0.6 -slice mcz 20
plot-waveforms -m 20 -td 0.022 -omega 3.8 -theta 8
test-minimization -n -mcz 20 -td 0.022 -resolution 51 151
```

Alternatively, run as Python modules (no installation needed):

```bash
python -m scripts.calc_mcz_vs_td -pbar -I 0.6 -slice mcz 20
python -m scripts.plot_waveforms -m 20 -td 0.022 -omega 3.8 -theta 8
```

## Authors
* Ben McKallip
* Tien Nguyen
* Tamanjyot Singh
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
