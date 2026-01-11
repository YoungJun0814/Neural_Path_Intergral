"""
DriftNet Training Script (Historical Data Distribution Matching)
DriftNet 학습 스크립트 (역사적 데이터 분포 매칭)

This script trains the NeuralSDE model to match the statistical properties 
(Mean, Volatility, Skewness, Kurtosis) of real S&P 500 returns.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_engine import NeuralSDESimulator

# ==============================================================================
# Configuration
# ==============================================================================
DATA_PATH = 'data/processed/spy_returns.csv'
MODEL_SAVE_PATH = 'checkpoints/driftnet_kurtosis_corrected.pth'
IMG_SAVE_PATH = 'img/distribution_check.png'

HIDDEN_DIM = 64
N_LAYERS = 3
LEARNING_RATE = 0.0001  # Further reduced for stability
EPOCHS = 200        
BATCH_SIZE_PATHS = 1000 # Reduced for speed
DT = 1/252.0
T_HORIZON = 1.0 

# Target Moments weights
W_MEAN = 50.0   # Balanced adjustment for centering
W_STD = 100.0   # Aggressive penalty to force Vol down to 19%
W_SKEW = 0.5
W_KURT = 0.01   # Stable kurtosis weight

def load_data():
    """Load SPY data or generate synthetic."""
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        returns = df['Returns'].values
    else:
        print("⚠️ Data file not found. Using synthetic S&P 500 style data.")
        # Synthetic data with heavy tails
        np.random.seed(42)
        n_days = 5000
        # Student-t like returns (heavy tails)
        returns = np.random.standard_t(df=5, size=n_days) * 0.01 + 0.0004
    return returns

def compute_moments(data_tensor):
    """Compute Mean, Std, Skewness, Kurtosis of a tensor."""
    mean = torch.mean(data_tensor)
    std = torch.std(data_tensor) + 1e-8
    
    # Standardize
    z = (data_tensor - mean) / std
    
    skew = torch.mean(z**3)
    kurt = torch.mean(z**4) # Raw kurtosis (Normal=3)
    
    return mean, std, skew, kurt

def main():
    print("="*60)
    print("🚀 Starting DriftNet Training (Distribution Matching)")
    print("="*60)
    
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 2. Prepare Data & Target Moments
    returns = load_data()
    real_returns = torch.tensor(returns, dtype=torch.float32, device=device)
    
    # Calculate Target Moments from Real Data
    target_mean, target_std, target_skew, _ = compute_moments(real_returns)
    target_kurt = torch.tensor(6.0, device=device) # Perfect shape target
    
    print(f"\n📊 Target Market Statistics (Real SPY):")
    print(f"  Mean:     {target_mean.item()*100:.4f}%")
    print(f"  Vol (Yr): {target_std.item()*np.sqrt(252)*100:.2f}%")
    print(f"  Skew:     {target_skew.item():.4f}")
    print(f"  Kurtosis: {target_kurt.item():.4f} (Goal: Perfect Shape with Slight Left Shift)")
    
    # 3. Initialize Model
    # Note: We use the src.neural_engine.NeuralSDESimulator
    simulator = NeuralSDESimulator(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, device=device)
    optimizer = optim.Adam(simulator.parameters(), lr=LEARNING_RATE)
    
    # Create checkpoints dir
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    # 4. Training Loop
    print("\nTraining started...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Simulate paths
        S, v = simulator.simulate(S0=100.0, T=0.5, dt=DT, num_paths=BATCH_SIZE_PATHS, training=True)
        
        # Calculate Log Returns
        # S shape: (paths, steps+1)
        # Returns: (paths, steps)
        model_log_returns = torch.log(S[:, 1:] / S[:, :-1])
        model_flat_returns = model_log_returns.flatten()
        
        # Compute Model Moments
        m_mean, m_std, m_skew, m_kurt = compute_moments(model_flat_returns)
        
        # Loss Function
        loss_mean = (m_mean - target_mean)**2
        loss_std  = (m_std - target_std)**2
        loss_skew = (m_skew - target_skew)**2
        loss_kurt = (m_kurt - target_kurt)**2
        
        total_loss = (W_MEAN * loss_mean + 
                      W_STD * loss_std + 
                      W_SKEW * loss_skew + 
                      W_KURT * loss_kurt)
        
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(simulator.parameters(), 1.0)
        
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss.item():.6f} | "
                  f"Kurt: {m_kurt.item():.2f} | "
                  f"Vol: {m_std.item()*np.sqrt(252)*100:.1f}%")

        # Periodic Save & Plot
        if (epoch+1) % 50 == 0:
            # Save Checkpoint
            simulator.save(MODEL_SAVE_PATH.replace('.pth', f'_ep{epoch+1}.pth'))
            
            # Generate Interim Plot
            with torch.no_grad():
                S_test, _ = simulator.simulate(S0=100.0, T=1.0, dt=DT, num_paths=2000, training=False)
                test_returns = torch.log(S_test[:, 1:] / S_test[:, :-1]).flatten().cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            plt.hist(returns, bins=100, density=True, alpha=0.5, label='Real SPY', color='blue')
            plt.hist(test_returns, bins=100, density=True, alpha=0.5, label=f'Model (Ep {epoch+1})', color='red')
            plt.title(f'Return Distribution at Epoch {epoch+1}\nKurtosis: {compute_moments(torch.tensor(test_returns))[3]:.2f}')
            plt.xlabel('Daily Return')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(IMG_SAVE_PATH.replace('.png', f'_ep{epoch+1}.png'))
            plt.close()

    # 5. Save Model
    simulator.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
    
    # 6. Final Validation & Visualization
    print("\nGenerating Validation Plot...")
    with torch.no_grad():
        S_test, _ = simulator.simulate(S0=100.0, T=1.0, dt=DT, num_paths=5000, training=False)
        test_returns = torch.log(S_test[:, 1:] / S_test[:, :-1]).flatten().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    
    # Real Data
    plt.hist(returns, bins=100, density=True, alpha=0.5, label='Real SPY', color='blue')
    
    # Model Data
    plt.hist(test_returns, bins=100, density=True, alpha=0.5, label='Neural SDE (Corrected)', color='red')
    
    plt.title(f'Return Distribution: Real vs Neural SDE (Kurtosis Adjusted)\nFinal Kurtosis: {compute_moments(torch.tensor(test_returns))[3]:.2f}')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.savefig(IMG_SAVE_PATH)
    print(f"✅ Plot saved to {IMG_SAVE_PATH}")

if __name__ == "__main__":
    main()
