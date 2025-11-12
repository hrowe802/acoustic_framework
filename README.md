# Acoustic Framework

An open-source framework for **speech acoustics analysis**, designed to streamline **automatic segmentation**, **acoustic feature extraction**, and **kinematic parameter computation** for speech and voice research.

Originally developed as a collaborative research tool, this framework provides modular scripts for integrating **Praat**, **MATLAB**, and **R** analyses into a unified Python-based workflow.

---

## Overview

The **Acoustic Framework** enables reproducible pipelines for analyzing speech recordings, segmenting utterances, and extracting a broad range of acoustic and kinematic features.

It supports both **automated pipelines** (for batch analysis) and **manual, research-driven** workflows.

This repository consolidates several acoustic analysis components under a clean, Pythonic package structure , making it easy to extend, test, and integrate into larger linguistic or clinical pipelines.

---

## Project Structure

```text
acoustic_framework/
  pipelines/                 # Processing pipelines and orchestration scripts
    ddk.py                   # DDK (Diadochokinetic) segmentation + analysis
  segmentation/              # Speech segmentation logic
    automatic_segmentation_ddk.py
  features/                  # Acoustic and kinematic feature extraction
    formant_formatting.py
    spectrum_formatting.py
    cepstrum_variables.py
    kinematic_variables.py
    prediction_formatting.py
  __init__.py
praat/                       # Praat scripts for spectral and timing analysis
r/                            # R scripts for statistical modeling
matlab/                       # MATLAB routines for feature extraction
notebooks/                    # Research notebooks and examples
docs/                         # Documentation and usage notes
scripts/                      # Utility and batch scripts
README.md
