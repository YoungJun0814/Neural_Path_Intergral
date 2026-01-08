
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.neural_engine import NeuralSDESimulator

# Configuration
DATA_PATH = 'data/processed/spy_returns.csv'
MODEL_PATH = 'checkpoints/driftnet_kurtosis_corrected.pth'
IMG_SAVE_PATH = 'img/distribution_check_fixed.png'
HIDDEN_DIM = 64
N_LAYERS = 3
DT = 1/252.0

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        returns = df['Returns'].values
    else:
        print("⚠️ Data file not found. Using synthetic.")
        np.random.seed(42)
        returns = np.random.standard_t(df=5, size=5000) * 0.01
    return returns

def compute_moments(data_tensor):
    mean = torch.mean(data_tensor)
    std = torch.std(data_tensor) + 1e-8
    z = (data_tensor - mean) / std
    skew = torch.mean(z**3)
    kurt = torch.mean(z**4)
    return mean, std, skew, kurt

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Load Data
    real_returns = load_data()
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    simulator = NeuralSDESimulator(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, device=device)
    simulator.load(MODEL_PATH)
    
    # 3. Generate Model Returns
    print("Generating paths...")
    with torch.no_grad():
        # Simulate enough paths for smooth hist
        S_test, _ = simulator.simulate(S0=100.0, T=1.0, dt=DT, num_paths=10000, training=False)
        model_returns = torch.log(S_test[:, 1:] / S_test[:, :-1]).flatten().cpu().numpy()

    # 4. Compute Stats
    real_t = torch.tensor(real_returns)
    model_t = torch.tensor(model_returns)
    
    r_stats = compute_moments(real_t)
    m_stats = compute_moments(model_t)
    
    print(f"\nStats Comparison:")
    print(f"Metric | Real SPY | DriftNet")
    print(f"-------|----------|---------")
    print(f"Vol    | {r_stats[1].item()*np.sqrt(252)*100:.2f}%   | {m_stats[1].item()*np.sqrt(252)*100:.2f}%")
    print(f"Kurt   | {r_stats[3].item():.2f}     | {m_stats[3].item():.2f}")

    # 5. Plot with SHARED BINS
    print("\nPlotting...")
    plt.figure(figsize=(10, 6))
    
    # Fixed shared bins for fair density comparison
    # Focus on the relevant range [-0.05, 0.05] (approx +/- 3 sigma)
    # This avoids outliers compressing the visual bins
    bins = np.linspace(-0.05, 0.05, 100)
    
    plt.hist(real_returns, bins=bins, density=True, alpha=0.5, label='Real SPY', color='blue')
    plt.hist(model_returns, bins=bins, density=True, alpha=0.5, label='DriftNet (Kurtosis Corrected)', color='red')
    
    plt.title(f'Return Distribution: Fixed Binning\nReal Kurt={r_stats[3]:.2f} vs Model Kurt={m_stats[3]:.2f}')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 0.05)
    
    plt.savefig(IMG_SAVE_PATH)
    print(f"✅ Corrected plot saved to {IMG_SAVE_PATH}")

if __name__ == "__main__":
    main()
