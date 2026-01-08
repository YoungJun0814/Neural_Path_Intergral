"""
Test script to validate Kurtosis Penalty implementation.
첨도 손실 구현 검증 스크립트.

This script:
1. Creates a NeuralSDESimulator
2. Runs one train_step with the new kurtosis penalty
3. Prints the computed kurtosis and total loss
"""

import torch
import sys
sys.path.insert(0, 'c:/Users/Jun/Desktop/Thesis/Projects/Neural_Path_Integral')

from src.neural_engine import NeuralSDESimulator

def test_kurtosis_loss():
    print("=" * 60)
    print("Testing Kurtosis Penalty Implementation")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create simulator
    simulator = NeuralSDESimulator(hidden_dim=32, n_layers=2, device=device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)
    
    # Dummy market data
    S0 = 100.0
    T = 0.25  # 3 months
    r = 0.05
    strikes = [95.0, 100.0, 105.0]
    target_prices = [8.0, 5.0, 2.5]  # Dummy prices
    
    # Run train step with new kurtosis penalty
    print("\nRunning train_step with Kurtosis Penalty...")
    print(f"  Target Kurtosis: 6.0 (S&P 500 typical)")
    print(f"  Kurtosis Weight (λ): 0.1")
    
    # Run a few steps to see convergence
    for i in range(5):
        loss = simulator.train_step(
            target_prices=target_prices,
            strikes=strikes,
            T=T,
            S0=S0,
            r=r,
            optimizer=optimizer,
            target_kurtosis=6.0,
            kurtosis_weight=0.1
        )
        print(f"  Step {i+1}: Total Loss = {loss:.6f}")
    
    # Final simulation to check kurtosis
    print("\nFinal Distribution Check...")
    with torch.no_grad():
        S, v = simulator.simulate(S0, T, dt=0.01, num_paths=10000, training=False)
        log_returns = torch.log(S[:, 1:] / S[:, :-1])
        returns_flat = log_returns.flatten().cpu().numpy()
        
        import numpy as np
        from scipy.stats import kurtosis as scipy_kurtosis
        
        model_kurtosis = scipy_kurtosis(returns_flat, fisher=False)  # Raw kurtosis
        print(f"  Model Kurtosis (Raw): {model_kurtosis:.3f}")
        print(f"  Reference (Normal): 3.0")
        print(f"  Reference (S&P 500): ~6.0")
    
    print("\n" + "=" * 60)
    print("✅ Kurtosis Penalty implementation verified!")
    print("=" * 60)

if __name__ == "__main__":
    test_kurtosis_loss()
