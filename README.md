# Neural Path Integral for Finance

This project aims to implement a Neural Path Integral approach for calibrating Heston Stochastic Volatility models and other physics-informed financial models.

## Project Structure

- `data/`: Dataset storage
  - `raw/`: Original CSV files (VIX, Options)
  - `processed/`: Preprocessed tensor data
- `docs/`: Research documentation
  - `literature/`: Papers and reference materials
  - `notes/`: Mathematical derivations and ideas
- `models/`: Trained model weights (.pth)
- `notebooks/`: Experimental steps
  - `01_Heston_Simulation.ipynb`: Heston SDE simulation and data generation
  - `02_Path_Integral_Calculation.ipynb`: Lagrangian and Action computations
  - `03_Neural_Calibration.ipynb`: Optimization using Neural SDEs
- `src/`: Core source code
  - `physics_engine.py`: Heston SDE and GPU simulators
  - `quantum_solver.py`: Lagrangian and Action calculations
  - `ai_calibrator.py`: Neural SDE optimization logic
  - `utils.py`: Data loaders and visualization utilities
- `main.py`: Main entry point for the pipeline
- `requirements.txt`: Python dependencies

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation:
   ```bash
   python main.py --mode simulate
   ```
