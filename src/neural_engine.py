"""
Neural SDE Engine / 신경망 확률 미분 방정식 엔진
==============================================

이 모듈은 딥러닝(Neural Network)을 사용하여 SDE(확률 미분 방정식)의
Drift(추세)와 Diffusion(확산) 함수를 학습하는 시뮬레이터를 구현합니다.

This module implements a simulator that uses deep learning (Neural Networks)
to learn the Drift and Diffusion functions of an SDE (Stochastic Differential Equation).

핵심 아이디어 / Key Idea:
    전통적 모델: dS = mu(theta) * dt + sigma(theta) * dW  (수식이 정해져 있음)
    Neural SDE:  dS = mu_net(S, v, t) * dt + sigma_net(S, v, t) * dW  (신경망이 학습)

참고 문헌 / References:
    - Kidger et al. (2021) "Neural SDEs as Infinite-Dimensional GANs"
    - Gierjatowicz et al. (2020) "Robust pricing and hedging via neural SDEs"
"""

import torch
import torch.nn as nn


class DriftNet(nn.Module):
    """
    Drift 신경망 / Drift Neural Network.
    
    SDE의 추세(Drift) 항 mu(S, v, t)를 근사합니다.
    Approximates the drift term mu(S, v, t) of the SDE.
    
    입력: (S_t, v_t, t) - 현재 주가, 변동성, 시간
    Input: (S_t, v_t, t) - Current price, volatility, time
    
    출력: 스칼라 (순간 드리프트)
    Output: Scalar (instantaneous drift)
    """
    
    def __init__(self, hidden_dim=64, n_layers=3):
        """
        Args:
            hidden_dim: 은닉층 차원 / Hidden layer dimension
            n_layers: 은닉층 수 / Number of hidden layers
        """
        super().__init__()
        
        layers = []
        # 입력층: (S, v, t, A) -> 4차원 / Input layer: 4 dimensions
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.SiLU())  # Swish 활성화 함수 / Swish activation
        
        # 은닉층 / Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # 출력층: 스칼라 / Output layer: scalar
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, S, v, t, avg_S=None):
        """
        순전파 / Forward pass.
        
        Args:
            S: 주가 텐서 (batch,) / Price tensor
            v: 변동성 텐서 (batch,) / Volatility tensor
            t: 시간 스칼라 / Time scalar
            avg_S: 이동 평균 텐서 (batch,) / Running Average (for Asian options)
        
        Returns:
            drift: 드리프트 값 (batch,) / Drift values
        """
        # 입력 정규화 (안정성 개선) / Normalize inputs for stability
        S_norm = torch.log(S + 1e-8)  # 로그 스케일 / Log scale
        v_norm = torch.log(v + 1e-8)  # 로그 스케일 / Log scale
        t_norm = t * torch.ones_like(S)  # 시간 브로드캐스트 / Broadcast time
        
        if avg_S is None:
            # If avg_S is missing (backward compatibility), assume it equals S (t=0 case)
            avg_S = S
            
        avg_S_norm = torch.log(avg_S + 1e-8)
        
        x = torch.stack([S_norm, v_norm, t_norm, avg_S_norm], dim=-1)  # (batch, 4)
        raw_output = self.net(x).squeeze(-1)  # (batch,)
        
        # Bound the control to avoid exploding gradients and model collapse
        # Range: [-1.0, 1.0] - Restored to 1.0 for High Precision Run: We need this strength for VR, will fix Bias with finer dt.
        return 1.0 * torch.tanh(raw_output)


class DiffNet(nn.Module):
    """
    Diffusion 신경망 / Diffusion Neural Network.
    
    SDE의 확산(Diffusion) 항 sigma(S, v, t)를 근사합니다.
    Approximates the diffusion term sigma(S, v, t) of the SDE.
    
    출력은 항상 양수가 되도록 Softplus를 사용합니다.
    Uses Softplus to ensure output is always positive.
    """
    
    def __init__(self, hidden_dim=64, n_layers=3):
        """
        Args:
            hidden_dim: 은닉층 차원 / Hidden layer dimension
            n_layers: 은닉층 수 / Number of hidden layers
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        # Softplus로 양수 보장 / Softplus ensures positivity
        layers.append(nn.Softplus())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, S, v, t):
        """
        순전파 / Forward pass.
        
        Returns:
            diffusion: 확산 값 (batch,), 항상 양수 / Diffusion values, always positive
        """
        S_norm = torch.log(S + 1e-8)
        v_norm = torch.log(v + 1e-8)
        t_norm = t * torch.ones_like(S)
        
        x = torch.stack([S_norm, v_norm, t_norm], dim=-1)
        return self.net(x).squeeze(-1)


class NeuralSDESimulator:
    """
    Neural SDE 시뮬레이터 / Neural SDE Simulator.
    
    Drift와 Diffusion 함수를 신경망으로 대체한 확률 미분 방정식 시뮬레이터입니다.
    A stochastic differential equation simulator where Drift and Diffusion
    functions are replaced by neural networks.
    
    동역학 / Dynamics:
        dS_t = DriftNet(S, v, t) * S_t * dt + DiffNet(S, v, t) * S_t * dW_t
    
    주의: 이 시뮬레이터는 학습(train) 후 사용해야 합니다.
    Note: This simulator must be trained before use.
    """
    
    def __init__(self, hidden_dim=64, n_layers=3, device='cuda'):
        """
        Args:
            hidden_dim: 신경망 은닉층 차원 / Neural network hidden dimension
            n_layers: 신경망 층 수 / Number of layers
            device: 연산 장치 / Computation device
        """
        self.device = device
        
        # 신경망 초기화 / Initialize neural networks
        self.drift_net = DriftNet(hidden_dim, n_layers).to(device)
        self.diff_net = DiffNet(hidden_dim, n_layers).to(device)
        
        # 변동성 초기값 (학습 시 고정 또는 학습 가능) / Initial vol (can be fixed or learned)
        self.v0 = 0.04  # 기본값 / Default value
    
    def simulate(self, S0, T, dt, num_paths, v0=None, training=False):
        """
        Neural SDE 모델로 가격 경로 시뮬레이션.
        Simulate price paths using Neural SDE model.
        
        Args:
            S0: 초기 주가 / Initial asset price
            T: 만기 시간 / Time to maturity
            dt: 시간 간격 / Time step size
            num_paths: 경로 수 / Number of paths
            v0: 초기 변동성 (선택) / Initial volatility (optional)
            training: 학습 모드 여부 / Training mode flag
        
        Returns:
            S: 가격 경로 (num_paths, n_steps+1) / Price paths
            v: 변동성 경로 (num_paths, n_steps+1) / Volatility paths (constant for now)
        """
        if v0 is None:
            v0 = self.v0
        
        n_steps = int(T / dt)
        
        # GPU 메모리 문제 방지 및 학습 안정성을 위해 training 모드에서는 inplace 연산 주의
        # For training stability, avoid inplace operations
        
        # 경로 초기화 (리스트 사용) / Initialize paths (use list)
        curr_S = torch.ones(num_paths, device=self.device) * S0
        curr_v = torch.ones(num_paths, device=self.device) * v0
        
        S_list = [curr_S]
        v_list = [curr_v]
        
        # 네트워크 모드 설정
        if training:
            self.drift_net.train()
            self.diff_net.train()
            context = torch.enable_grad()
        else:
            self.drift_net.eval()
            self.diff_net.eval()
            context = torch.no_grad()
            
        with context:
            sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
            
            for t_idx in range(1, n_steps + 1):
                t = (t_idx - 1) * dt
                
                # 브라운 운동 증분 / Brownian motion increment
                dW = torch.randn(num_paths, device=self.device) * sqrt_dt
                
                # Neural Drift & Diffusion
                mu = self.drift_net(curr_S, curr_v, t)
                sigma = self.diff_net(curr_S, curr_v, t)
                
                # Euler-Maruyama 이산화 (In-place 연산 피하기)
                # Euler-Maruyama discretization (Avoid in-place ops)
                # dS = mu * S * dt + sigma * S * dW
                dS = mu * curr_S * dt + sigma * curr_S * dW
                
                # Update state
                next_S = curr_S + dS
                
                # 주가 0 이하 방지 (ReLU or Softplus)
                next_S = torch.nn.functional.relu(next_S) + 1e-8
                
                S_list.append(next_S)
                v_list.append(curr_v) 
                
                curr_S = next_S
        
        # Stack results
        S = torch.stack(S_list, dim=1)
        v = torch.stack(v_list, dim=1)
        
        return S, v
    
    def train_step(self, target_prices, strikes, T, S0, r, optimizer):
        """
        단일 학습 스텝 (옵션 가격 매칭).
        Single training step (option price matching).
        
        Monte Carlo 시뮬레이션으로 옵션 가격을 생성하고,
        시장 가격과의 차이(MSE)를 최소화합니다.
        
        Args:
            target_prices: 시장 옵션 가격 (타겟) / Market option prices (target)
            strikes: 행사가 배열 / Strike prices
            T: 만기 / Time to maturity
            S0: 초기 주가 / Initial price
            r: 무위험 이자율 / Risk-free rate
            optimizer: PyTorch 옵티마이저 / PyTorch optimizer
        
        Returns:
            loss: 손실값 / Loss value
        """
        self.drift_net.train()
        self.diff_net.train()
        
        optimizer.zero_grad()
        
        # 시뮬레이션 (경로 수 줄여서 학습 속도 향상)
        # Simulate (reduce paths for faster training)
        num_paths = 2000
        # dt가 만기보다 길면 루프가 안 돌아서 gradient가 끊김. 동적 dt 설정.
        # Use dynamic dt to ensure simulation runs even for short maturities.
        dt = min(0.01, T / 3.0) 
        
        S, _ = self.simulate(S0, T, dt, num_paths, training=True)
        S_final = S[:, -1]  # 만기 시점 주가 / Price at maturity
        
        # 콜옵션 페이오프 계산 / Compute call payoffs
        strikes_gpu = torch.tensor(strikes, device=self.device, dtype=torch.float32)
        payoffs = torch.maximum(
            S_final.unsqueeze(1) - strikes_gpu.unsqueeze(0),
            torch.tensor(0.0, device=self.device)
        )  # (num_paths, num_strikes)
        
        # 할인된 평균 가격 / Discounted average price
        model_prices = torch.mean(payoffs, dim=0) * torch.exp(torch.tensor(-r * T, device=self.device))
        
        # MSE 손실 / MSE loss
        target_gpu = torch.tensor(target_prices, device=self.device, dtype=torch.float32)
        loss = torch.mean((model_prices - target_gpu) ** 2)
        
        # 역전파 / Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """
        모델 저장 / Save model.
        """
        torch.save({
            'drift_net': self.drift_net.state_dict(),
            'diff_net': self.diff_net.state_dict(),
            'v0': self.v0
        }, path)
    
        checkpoint = torch.load(path, map_location=self.device)
        self.drift_net.load_state_dict(checkpoint['drift_net'])
        self.diff_net.load_state_dict(checkpoint['diff_net'])
        self.v0 = checkpoint.get('v0', 0.04)


class NeuralImportanceSampler:
    """
    Neural Importance Sampling for Efficient Option Pricing.
    
    Learns the optimal drift (control force) to minimize the variance
    of the Payoff Estimator.
    """
    def __init__(self, simulator, hidden_dim=64, n_layers=3):
        """
        Args:
            simulator: src.physics_engine.MarketSimulator
        """
        self.sim = simulator
        self.device = simulator.device
        
        # Policy Network (Drift Control)
        # Input: (S, v, t) -> Output: u (control force, scalar or vector)
        # Reusing DriftNet architecture
        self.control_net = DriftNet(hidden_dim, n_layers).to(self.device)
        
        # Initialize weights to near-zero to start with uniform (Standard MC) behavior
        for p in self.control_net.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.001)
        
    def get_control_fn(self):
        """
        Returns a callable function appropriate for simulate_controlled.
        control_fn(t, S, v, A=None) -> u
        """
        def control_fn(t, S, v, A=None):
            # DriftNet expects (S, v, t, A)
            return self.control_net(S, v, t, A)
        return control_fn

    def train_step(self, S0, K, T, r, barrier_level, barrier_type, optimizer, num_paths=1000):
        """
        Train the control network to minimize variance of the price estimator.
        Loss = E[ (Payoff * LikelihoodRatio)^2 ] (Second Moment)
        """
        self.control_net.train()
        optimizer.zero_grad()
        
        # 1. Controlled Simulation
        dt = min(0.01, T / 100.0)
        control_fn = self.get_control_fn()
        
        # We need gradients to flow through 'simulate_controlled' back to 'control_net'
        # The variables S, v depend on u.
        # But wait, simulate_controlled implementation iterates in Python loop.
        # PyTorch autograd handles unrolling, so gradients should flow.
        
        S_paths, _, log_weights, barrier_hit, _ = self.sim.simulate_controlled(
            S0=S0, v0=self.sim.theta, T=T, dt=dt, num_paths=num_paths,
            control_fn=control_fn, barrier_level=barrier_level, barrier_type=barrier_type
        )
        
        # 2. Payoff Calculation
        S_final = S_paths[:, -1]
        
        # Put Option Payoff
        payoffs = torch.maximum(torch.tensor(K, device=self.device) - S_final, 
                              torch.tensor(0.0, device=self.device))
        
        if barrier_hit is not None:
             payoffs = payoffs * (~barrier_hit).float()
             
        # 3. Reweighted Payoff
        # Z = Payoff * exp(log_weight)
        # We want to minimize Var[Z] ~ E[Z^2]
        
        # Note regarding gradient detachment:
        # We want to optimize theta (params of u).
        # u affects both 'payoffs' (via S_final) and 'log_weights'.
        # Standard REINFORCE or Pathwise Derivative?
        # Here we are using Pathwise Derivative (reparameterization trick is implicit in simulate).
        # Everything is differentiable.
        
        weighted_payoffs = payoffs * torch.exp(log_weights)
        
        # Loss: Second Moment (minimizing this minimizes variance)
        loss = torch.mean(weighted_payoffs ** 2)
        
        loss.backward()
        optimizer.step()
        
        return loss.item(), torch.mean(weighted_payoffs).item()
        
    def save(self, path):
        torch.save(self.control_net.state_dict(), path)
        
    def load(self, path):
        self.control_net.load_state_dict(torch.load(path, map_location=self.device))
