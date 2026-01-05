import numpy as np
import matplotlib.pyplot as plt
import os

def generate_concept_plot():
    # Heston Parametrs
    S0 = 100.0
    v0 = 0.04
    mu = 0.05
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    
    T = 1.0
    dt = 1/252
    steps = int(T/dt)
    t_axis = np.linspace(0, T, steps+1)
    
    # ----------------------------------------------------
    # 1. Simulate Normal Path (Monte Carlo - Pure Numpy)
    # ----------------------------------------------------
    np.random.seed(42) # Fixed seed for reproducibility
    S_normal = [S0]
    v_normal = [v0]
    
    curr_S = S0
    curr_v = v0
    
    for i in range(steps):
        # Correlated Brownian Motions
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        dW_S = Z1 * np.sqrt(dt)
        dW_v = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
        
        # Volatility Process (Heston)
        dv = kappa * (theta - curr_v) * dt + xi * np.sqrt(max(curr_v, 0)) * dW_v
        curr_v = max(curr_v + dv, 1e-4) # Reflecting/Full truncation
        
        # Price Process
        dS = mu * curr_S * dt + np.sqrt(max(curr_v, 0)) * curr_S * dW_S
        curr_S = max(curr_S + dS, 1e-2)
        
        S_normal.append(curr_S)
        v_normal.append(curr_v)

    # ----------------------------------------------------
    # 2. Simulate "Controlled" Path (Concept)
    # ----------------------------------------------------
    # We artificially add a negative component to the drift
    # to simulate the DriftNet controller's effect.
    S_crash = [S0]
    v_crash = [v0]
    
    curr_S = S0
    curr_v = v0
    
    # We reuse the seed logic roughly to show "similar market conditions" but differing outcome
    # Or we can just use the same seed to show perfectly the effect of control vs no control
    np.random.seed(42) 
    
    for i in range(steps):
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        dW_S = Z1 * np.sqrt(dt)
        dW_v = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
        
        # Volatility Process (Same dynamics)
        dv = kappa * (theta - curr_v) * dt + xi * np.sqrt(max(curr_v, 0)) * dW_v
        curr_v = max(curr_v + dv, 1e-4)
        
        # Controlled Price Process
        # Control u(t) becomes more aggressive as time passes or condition worsens
        # Concept: "DriftNet pushes price down aggressively"
        
        # Simple polynomial control for visualization
        time_factor = (i / steps)**2
        control_u = -1.5 * time_factor - 0.2
        
        # Modified Drift: mu + control_u
        dS = (mu + control_u) * curr_S * dt + np.sqrt(max(curr_v, 0)) * curr_S * dW_S
        curr_S = max(curr_S + dS, 1e-2)
        
        S_crash.append(curr_S)
        v_crash.append(curr_v)

    # ----------------------------------------------------
    # Plotting
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plot Normal Path
    plt.plot(t_axis, S_normal, label='Baseline Market (Uncontrolled)', color='gray', alpha=0.6, linestyle='--', linewidth=1.5)
    
    # Plot Controlled Path
    plt.plot(t_axis, S_crash, label='DriftNet Controlled (Crash Generated)', color='#e74c3c', linewidth=3)
    
    # Threshold Line
    plt.axhline(y=100, color='black', linestyle=':', alpha=0.3)
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Crash Threshold (K=80)')
    
    # Annotations
    plt.title('DriftNet Concept: Forcing Black Swan Events', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Year)', fontsize=12)
    plt.ylabel('Asset Price ($)', fontsize=12)
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, alpha=0.2)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save
    if not os.path.exists('img'):
        os.makedirs('img')
    output_path = 'img/concept_crash.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to {output_path}")

if __name__ == "__main__":
    generate_concept_plot()
