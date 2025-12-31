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
        # 입력층: (S, v, t) -> 3차원 / Input layer: 3 dimensions
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.SiLU())  # Swish 활성화 함수 / Swish activation
        
        # 은닉층 / Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # 출력층: 스칼라 / Output layer: scalar
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, S, v, t):
        """
        순전파 / Forward pass.
        
        Args:
            S: 주가 텐서 (batch,) / Price tensor
            v: 변동성 텐서 (batch,) / Volatility tensor
            t: 시간 스칼라 / Time scalar
        
        Returns:
            drift: 드리프트 값 (batch,) / Drift values
        """
        # 입력 정규화 (안정성 개선) / Normalize inputs for stability
        S_norm = torch.log(S + 1e-8)  # 로그 스케일 / Log scale
        v_norm = torch.log(v + 1e-8)  # 로그 스케일 / Log scale
        t_norm = t * torch.ones_like(S)  # 시간 브로드캐스트 / Broadcast time
        
        x = torch.stack([S_norm, v_norm, t_norm], dim=-1)  # (batch, 3)
        return self.net(x).squeeze(-1)  # (batch,)


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
    
    def simulate(self, S0, T, dt, num_paths, v0=None):
        """
        Neural SDE 모델로 가격 경로 시뮬레이션 (추론 모드).
        Simulate price paths using Neural SDE model (inference mode).
        
        Args:
            S0: 초기 주가 / Initial asset price
            T: 만기 시간 / Time to maturity
            dt: 시간 간격 / Time step size
            num_paths: 경로 수 / Number of paths
            v0: 초기 변동성 (선택) / Initial volatility (optional)
        
        Returns:
            S: 가격 경로 (num_paths, n_steps+1) / Price paths
            v: 변동성 경로 (num_paths, n_steps+1) / Volatility paths (constant for now)
        """
        if v0 is None:
            v0 = self.v0
        
        n_steps = int(T / dt)
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        # 경로 초기화 / Initialize paths
        S = torch.zeros(num_paths, n_steps + 1, device=self.device)
        v = torch.ones(num_paths, n_steps + 1, device=self.device) * v0
        S[:, 0] = S0
        
        # 추론 모드 / Inference mode
        self.drift_net.eval()
        self.diff_net.eval()
        
        with torch.no_grad():
            for t_idx in range(1, n_steps + 1):
                t = t_idx * dt
                S_prev = S[:, t_idx - 1]
                v_prev = v[:, t_idx - 1]
                
                # 브라운 운동 증분 / Brownian motion increment
                dW = torch.randn(num_paths, device=self.device) * sqrt_dt
                
                # Neural Drift & Diffusion
                mu = self.drift_net(S_prev, v_prev, t)
                sigma = self.diff_net(S_prev, v_prev, t)
                
                # Euler-Maruyama 이산화 / Euler-Maruyama discretization
                # dS = mu * S * dt + sigma * S * dW
                dS = mu * S_prev * dt + sigma * S_prev * dW
                
                S[:, t_idx] = torch.clamp(S_prev + dS, min=1e-8)
        
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
        dt = 0.01
        S, _ = self.simulate(S0, T, dt, num_paths)
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
    
    def load(self, path):
        """
        모델 로드 / Load model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.drift_net.load_state_dict(checkpoint['drift_net'])
        self.diff_net.load_state_dict(checkpoint['diff_net'])
        self.v0 = checkpoint.get('v0', 0.04)
