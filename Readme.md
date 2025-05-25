# AIMEC NPZ Surrogate Model

This repository contains code and analysis for developing surrogate models of the Nutrient-Phytoplankton-Zooplankton-Detritus (NPZ-D) ecosystem using machine learning techniques, created as part of an application for the JAMSTEC AIMEC Researcher position. The project integrates synthetic oceanographic data (Sea Surface Temperature and chlorophyll) with an enhanced NPZ model and compares Random Forest and PyTorch Neural Network surrogate models.

## Overview
- **Goal**: Build and compare Random Forest and PyTorch Neural Network surrogates to approximate the NPZ-D model dynamics.
- **Data**: Synthetic data generated based on realistic oceanographic forcing conditions.
- **Tools**: Python, scikit-learn, PyTorch, Jupyter Notebook, pandas, numpy, matplotlib.

## Files
- `scripts/generate_training_data.py`: Generates synthetic training data (`real_forced_training_data.csv`) using an enhanced NPZ model with real forcing conditions.
- `scripts/random_forest_surrogate.py`: Trains a Random Forest model and saves predictions to `rf_predictions.csv`.
- `scripts/neural_network_surrogate_pytorch.py`: Trains a PyTorch neural network model and saves predictions to `nn_predictions_pytorch.csv`.
- `scripts/compare_models.py`: Compares the performance of Random Forest and Neural Network models and generates comparison plots.
- `notebooks/test.ipynb`: Visualizes model predictions and performance metrics.
- `figures/`: Contains generated plots (e.g., `chlorophyll_distribution.png`, `prediction_scatter_plots.png`, `model_comparison_scatter_plots.png`).

## Installation
To run this project, install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn torch jupyter