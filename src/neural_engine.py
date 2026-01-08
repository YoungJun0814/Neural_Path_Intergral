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


class VolNet(nn.Module):
    """
    Volatility Dynamics Neural Network / 변동성 동역학 신경망.
    
    SDE의 분산 과정 dv = a(S, v, t) dt + b(S, v, t) dW_v를 근사합니다.
    Approximates the variance process: dv = a(S, v, t) dt + b(S, v, t) dW_v.
    
    핵심: Heston 모델의 mean-reversion 특성을 학습하면서도
    실제 데이터에서 나타나는 비선형적인 Vol dynamics를 포착합니다.
    
    Key: Learns mean-reversion like Heston while capturing
    nonlinear vol dynamics from real data.
    """
    
    def __init__(self, hidden_dim=64, n_layers=3):
        """
        Args:
            hidden_dim: 은닉층 차원 / Hidden layer dimension
            n_layers: 은닉층 수 / Number of hidden layers
        """
        super().__init__()
        
        # Drift Network for Volatility (a in dv = a*dt + b*dW)
        # 변동성 Drift 네트워크
        drift_layers = []
        drift_layers.append(nn.Linear(3, hidden_dim))  # Input: (S, v, t)
        drift_layers.append(nn.SiLU())
        
        for _ in range(n_layers - 1):
            drift_layers.append(nn.Linear(hidden_dim, hidden_dim))
            drift_layers.append(nn.SiLU())
        
        drift_layers.append(nn.Linear(hidden_dim, 1))
        self.drift_net = nn.Sequential(*drift_layers)
        
        # Diffusion Network for Volatility (b in dv = a*dt + b*dW)
        # 변동성 Diffusion 네트워크 (항상 양수)
        diff_layers = []
        diff_layers.append(nn.Linear(3, hidden_dim))
        diff_layers.append(nn.SiLU())
        
        for _ in range(n_layers - 1):
            diff_layers.append(nn.Linear(hidden_dim, hidden_dim))
            diff_layers.append(nn.SiLU())
        
        diff_layers.append(nn.Linear(hidden_dim, 1))
        diff_layers.append(nn.Softplus())  # 양수 보장 / Ensure positivity
        self.diff_net = nn.Sequential(*diff_layers)
        
        # Learnable mean-reversion parameters (prior from Heston)
        # 학습 가능한 평균 회귀 파라미터 (Heston 사전 분포)
        self.kappa = nn.Parameter(torch.tensor(2.0))  # Mean reversion speed
        self.theta = nn.Parameter(torch.tensor(0.04))  # Long-term variance
    
    def forward(self, S, v, t):
        """
        Returns both drift and diffusion for volatility process.
        변동성 과정의 drift와 diffusion을 모두 반환합니다.
        
        Args:
            S: 주가 텐서 (batch,) / Price tensor
            v: 변동성 텐서 (batch,) / Volatility tensor
            t: 시간 스칼라 / Time scalar
        
        Returns:
            a: 변동성 drift (batch,) / Volatility drift
            b: 변동성 diffusion (batch,) / Volatility diffusion
        """
        # 입력 정규화 / Normalize inputs
        S_norm = torch.log(S + 1e-8)
        v_norm = torch.log(v + 1e-8)
        t_norm = t * torch.ones_like(S)
        
        x = torch.stack([S_norm, v_norm, t_norm], dim=-1)
        
        # Neural correction to Heston mean-reversion
        # Heston 평균 회귀에 대한 신경망 보정
        # a = kappa * (theta - v) + neural_correction
        heston_drift = self.kappa * (torch.clamp(self.theta, min=1e-4) - v)
        neural_correction = self.drift_net(x).squeeze(-1) * 0.1  # Scale down correction
        a = heston_drift + neural_correction
        
        # b = neural_diffusion (replaces xi * sqrt(v) in Heston)
        # Heston의 xi * sqrt(v)를 대체
        b = self.diff_net(x).squeeze(-1) * torch.sqrt(torch.clamp(v, min=1e-8))
        
        return a, b


class NeuralSDESimulator:
    """
    Neural SDE 시뮬레이터 (Stochastic Volatility 버전) / Neural SDE Simulator (Stochastic Vol).
    
    이 시뮬레이터는 가격(S)과 변동성(v) 모두를 확률적으로 진화시킵니다.
    This simulator evolves BOTH price (S) and volatility (v) stochastically.
    
    동역학 / Dynamics:
        dS_t = mu(S, v, t) * S_t * dt + sigma(S, v, t) * S_t * dW_S
        dv_t = a(S, v, t) * dt + b(S, v, t) * dW_v
        
        with Corr(dW_S, dW_v) = rho (학습 가능 / Learnable)
    
    핵심 개선사항 / Key Improvements:
        - 변동성이 더 이상 상수가 아닙니다 / Volatility is no longer constant
        - S-v 상관관계를 학습합니다 / Learns S-v correlation (leverage effect)
        - Heston 사전 분포 + 신경망 보정 / Heston prior + Neural corrections
    """
    
    def __init__(self, hidden_dim=64, n_layers=3, device='cuda', rho_init=-0.7):
        """
        Args:
            hidden_dim: 신경망 은닉층 차원 / Neural network hidden dimension
            n_layers: 신경망 층 수 / Number of layers
            device: 연산 장치 / Computation device
            rho_init: 초기 상관계수 / Initial S-v correlation (typically negative)
        """
        self.device = device
        
        # =====================================================================
        # Neural Networks for Price Dynamics / 가격 동역학 신경망
        # =====================================================================
        self.drift_net = DriftNet(hidden_dim, n_layers).to(device)
        self.diff_net = DiffNet(hidden_dim, n_layers).to(device)
        
        # =====================================================================
        # Neural Network for Volatility Dynamics / 변동성 동역학 신경망 [NEW]
        # =====================================================================
        self.vol_net = VolNet(hidden_dim, n_layers).to(device)
        
        # =====================================================================
        # Learnable Correlation Parameter / 학습 가능한 상관계수 [NEW]
        # =====================================================================
        # Rho는 보통 음수 (leverage effect: 가격 하락 시 변동성 상승)
        # Rho is typically negative (leverage effect: price drops -> vol rises)
        rho_raw_init = self._inverse_tanh(rho_init)
        self._rho_raw = torch.nn.Parameter(
            torch.tensor(rho_raw_init, device=device, dtype=torch.float32)
        )
        
        # 변동성 초기값 / Initial volatility
        self.v0 = 0.04
    
    @staticmethod
    def _inverse_tanh(y):
        """Inverse of tanh for parameter initialization."""
        y = max(-0.999, min(0.999, y))  # Clamp to valid range
        return 0.5 * torch.log(torch.tensor((1 + y) / (1 - y)))
    
    @property
    def rho(self):
        """Get rho in [-1, 1] range via tanh transformation."""
        return torch.tanh(self._rho_raw)
    
    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.drift_net.parameters())
        params += list(self.diff_net.parameters())
        params += list(self.vol_net.parameters())
        params += [self._rho_raw]
        return params
    
    def simulate(self, S0, T, dt, num_paths, v0=None, training=False):
        """
        Stochastic Volatility Neural SDE 시뮬레이션.
        Simulate paths with STOCHASTIC volatility.
        
        Args:
            S0: 초기 주가 / Initial asset price
            T: 만기 시간 / Time to maturity
            dt: 시간 간격 / Time step size
            num_paths: 경로 수 / Number of paths
            v0: 초기 변동성 (선택) / Initial volatility (optional)
            training: 학습 모드 여부 / Training mode flag
        
        Returns:
            S: 가격 경로 (num_paths, n_steps+1) / Price paths
            v: 변동성 경로 (num_paths, n_steps+1) / Volatility paths (NOW STOCHASTIC!)
        """
        if v0 is None:
            v0 = self.v0
        
        n_steps = max(1, int(T / dt))
        
        # 경로 초기화 / Initialize paths
        curr_S = torch.ones(num_paths, device=self.device) * S0
        curr_v = torch.ones(num_paths, device=self.device) * v0
        
        S_list = [curr_S]
        v_list = [curr_v]
        
        # 네트워크 모드 설정 / Set network mode
        if training:
            self.drift_net.train()
            self.diff_net.train()
            self.vol_net.train()
            context = torch.enable_grad()
        else:
            self.drift_net.eval()
            self.diff_net.eval()
            self.vol_net.eval()
            context = torch.no_grad()
        
        with context:
            sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
            rho = self.rho  # Get current rho value
            sqrt_one_minus_rho2 = torch.sqrt(1 - rho ** 2 + 1e-8)
            
            for t_idx in range(1, n_steps + 1):
                t = (t_idx - 1) * dt
                
                # =============================================================
                # Correlated Brownian Motions / 상관된 브라운 운동 [NEW]
                # =============================================================
                z1 = torch.randn(num_paths, device=self.device)  # For S
                z2 = torch.randn(num_paths, device=self.device)  # For v (independent)
                
                dW_S = z1 * sqrt_dt
                dW_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt  # Correlated
                
                # =============================================================
                # Price Dynamics / 가격 동역학
                # =============================================================
                mu = self.drift_net(curr_S, curr_v, t)
                sigma = self.diff_net(curr_S, curr_v, t)
                
                dS = mu * curr_S * dt + sigma * curr_S * dW_S
                next_S = torch.clamp(curr_S + dS, min=1e-8)
                
                # =============================================================
                # Volatility Dynamics / 변동성 동역학 [NEW - Key Fix!]
                # =============================================================
                a, b = self.vol_net(curr_S, curr_v, t)  # Get vol drift and diffusion
                
                dv = a * dt + b * dW_v
                next_v = torch.clamp(curr_v + dv, min=1e-8)  # Ensure v > 0
                
                # Append to lists
                S_list.append(next_S)
                v_list.append(next_v)  # NOW EVOLVING! / 이제 진화합니다!
                
                # Update state
                curr_S = next_S
                curr_v = next_v  # [CRITICAL FIX] / 핵심 수정!
        
        # Stack results
        S = torch.stack(S_list, dim=1)
        v = torch.stack(v_list, dim=1)
        
        return S, v

    def simulate_controlled(self, S0, T, dt, num_paths, control_net, v0=None):
        """
        NPI를 위한 제어된 시뮬레이션 (Controlled Simulation).
        Simulates controlled paths for Neural Path Integral.
        
        Dynamics under Control:
            dS = (mu + sigma * u) * S * dt + sigma * S * dW_S
            
        Args:
            control_net: ControlForce Network (Input: S, v, t -> Output: u)
            
        Returns:
            S: Controlled Price paths
            log_weights: Girsanov log-weights (log dP/dQ)
        """
        if v0 is None:
            v0 = self.v0
        
        n_steps = max(1, int(T / dt))
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        rho = self.rho
        sqrt_one_minus_rho2 = torch.sqrt(1 - rho ** 2 + 1e-8)
        
        # Paths
        curr_S = torch.ones(num_paths, device=self.device) * S0
        curr_v = torch.ones(num_paths, device=self.device) * v0
        
        S_list = [curr_S]
        
        # Girsanov Accumulators
        log_weight = torch.zeros(num_paths, device=self.device)
        
        # Eval mode for all networks
        self.drift_net.eval()
        self.diff_net.eval()
        self.vol_net.eval()
        if hasattr(control_net, 'eval'):
            control_net.eval()
            
        with torch.no_grad():
            for t_idx in range(1, n_steps + 1):
                t = (t_idx - 1) * dt
                
                # 1. Noise Generation
                z1 = torch.randn(num_paths, device=self.device)
                z2 = torch.randn(num_paths, device=self.device)
                
                dW_S = z1 * sqrt_dt
                dW_v = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt
                
                # 2. Get Model Parameters
                mu = self.drift_net(curr_S, curr_v, t)
                sigma = self.diff_net(curr_S, curr_v, t)
                a, b = self.vol_net(curr_S, curr_v, t)
                
                # 3. Get Control Force u(S, v, t)
                # Note: control_net might expect (S, v, t) or (S, v, t, avg_S)
                # Assuming standard DriftNet signature (S, v, t, avg_S=None)
                u = control_net(curr_S, curr_v, t)
                
                # 4. Evolve S (Controlled)
                # dS = (mu + sigma*u)dt + sigma*dW
                # Here we apply u to the drift part.
                # Effective drift = mu + sigma * u
                drift_S = (mu + sigma * u) * curr_S * dt
                diff_S = sigma * curr_S * dW_S
                
                dS = drift_S + diff_S
                next_S = torch.clamp(curr_S + dS, min=1e-8)
                
                # 5. Evolve v (Standard)
                # Note: If we interpret control as changing Measure, v drift might change too
                # For simplicity in NPI v1, we assume control only affects Price Drift formulation directly
                # or that u is orthogonal to v noise. But here they are correlated.
                # Proper Girsanov would imply: z1 -> z1 + u*sqrt(dt)
                # Then dW_v depends on z1, so dW_v also has a drift shift of rho*u*dt.
                # Let's implement FULL Girsanov consistency:
                # The simulation above uses z1. If z1 is "shifted noise", then
                # dW_v includes rho * (z1) * sqrt(dt). 
                # Meaning v is also affected by the control u via correlation.
                
                dv = a * dt + b * dW_v
                next_v = torch.clamp(curr_v + dv, min=1e-8)
                
                # 6. Accumulate Girsanov Weight
                # log L = - int u * dW - 0.5 int u^2 dt
                # Here dW is the Brownian driver of S (z1).
                log_weight = log_weight - u * z1 * sqrt_dt - 0.5 * (u**2) * dt
                
                curr_S = next_S
                curr_v = next_v
                S_list.append(curr_S)
                
        S = torch.stack(S_list, dim=1)
        return S, log_weight

    def train_step(self, target_prices, strikes, T, S0, r, optimizer, 
                    target_kurtosis=6.0, kurtosis_weight=0.1):
        """
        단일 학습 스텝 (옵션 가격 매칭 + 첨도 제약).
        Single training step (option price matching + kurtosis penalty).
        
        Monte Carlo 시뮬레이션으로 옵션 가격을 생성하고,
        시장 가격과의 차이(MSE)를 최소화합니다.
        추가로, 수익률 분포의 첨도(Kurtosis)가 실제 시장과 유사하도록 학습합니다.
        
        Args:
            target_prices: 시장 옵션 가격 (타겟) / Market option prices (target)
            strikes: 행사가 배열 / Strike prices
            T: 만기 / Time to maturity
            S0: 초기 주가 / Initial price
            r: 무위험 이자율 / Risk-free rate
            optimizer: PyTorch 옵티마이저 / PyTorch optimizer
            target_kurtosis: 목표 첨도 (S&P 500 일간 수익률 ~6.0) / Target kurtosis
            kurtosis_weight: 첨도 손실 가중치 (λ) / Kurtosis loss weight
        
        Returns:
            loss: 손실값 / Loss value
        """
        # Set all networks to training mode / 모든 네트워크를 학습 모드로 설정
        self.drift_net.train()
        self.diff_net.train()
        self.vol_net.train()  # [NEW] 변동성 네트워크도 학습 모드로!
        
        optimizer.zero_grad()
        
        # 시뮬레이션 (경로 수 줄여서 학습 속도 향상)
        # Simulate (reduce paths for faster training)
        num_paths = 2000
        # dt가 만기보다 길면 루프가 안 돌아서 gradient가 끊김. 동적 dt 설정.
        # Use dynamic dt to ensure simulation runs even for short maturities.
        dt = min(0.01, T / 3.0) 
        
        S, v = self.simulate(S0, T, dt, num_paths, training=True)
        S_final = S[:, -1]  # 만기 시점 주가 / Price at maturity
        
        # 콜옵션 페이오프 계산 / Compute call payoffs
        strikes_gpu = torch.tensor(strikes, device=self.device, dtype=torch.float32)
        payoffs = torch.maximum(
            S_final.unsqueeze(1) - strikes_gpu.unsqueeze(0),
            torch.tensor(0.0, device=self.device)
        )  # (num_paths, num_strikes)
        
        # 할인된 평균 가격 / Discounted average price
        model_prices = torch.mean(payoffs, dim=0) * torch.exp(torch.tensor(-r * T, device=self.device))
        
        # =====================================================================
        # [NEW] 첨도 손실 (Kurtosis Penalty) / Kurtosis Loss
        # =====================================================================
        # 일간 수익률 계산 / Calculate daily returns
        # S shape: (num_paths, n_steps+1)
        log_returns = torch.log(S[:, 1:] / S[:, :-1])  # (num_paths, n_steps)
        returns_flat = log_returns.flatten()  # Flatten for kurtosis calculation
        
        # 첨도 계산 (Fisher's definition: Normal = 3) / Compute kurtosis
        mean_r = torch.mean(returns_flat)
        std_r = torch.std(returns_flat) + 1e-8  # Avoid division by zero
        z = (returns_flat - mean_r) / std_r
        model_kurtosis = torch.mean(z ** 4)  # Raw kurtosis (Normal = 3)
        
        # 첨도 손실: 목표 첨도와의 차이 / Kurtosis penalty
        kurtosis_loss = (model_kurtosis - target_kurtosis) ** 2
        
        # =====================================================================
        # 총 손실 = 가격 MSE + λ * 첨도 손실 / Total Loss
        # =====================================================================
        target_gpu = torch.tensor(target_prices, device=self.device, dtype=torch.float32)
        price_loss = torch.mean((model_prices - target_gpu) ** 2)
        
        loss = price_loss + kurtosis_weight * kurtosis_loss
        
        # 역전파 / Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """
        모델 저장 / Save model.
        
        Saves all networks, rho parameter, and v0.
        모든 네트워크, rho 파라미터, v0를 저장합니다.
        """
        torch.save({
            'drift_net': self.drift_net.state_dict(),
            'diff_net': self.diff_net.state_dict(),
            'vol_net': self.vol_net.state_dict(),  # [NEW]
            '_rho_raw': self._rho_raw.data,  # [NEW]
            'v0': self.v0
        }, path)
    
    def load(self, path):
        """
        모델 로드 / Load model.
        
        Loads all networks, rho parameter, and v0.
        모든 네트워크, rho 파라미터, v0를 로드합니다.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.drift_net.load_state_dict(checkpoint['drift_net'])
        self.diff_net.load_state_dict(checkpoint['diff_net'])
        
        # Load vol_net if present (backward compatibility)
        if 'vol_net' in checkpoint:
            self.vol_net.load_state_dict(checkpoint['vol_net'])
        
        # Load rho if present (backward compatibility)
        if '_rho_raw' in checkpoint:
            self._rho_raw.data = checkpoint['_rho_raw']
        
        self.v0 = checkpoint.get('v0', 0.04)


class NeuralImportanceSampler:
    """
    Neural Importance Sampling for Efficient Option Pricing.
    효율적인 옵션 가격 결정을 위한 신경망 중요도 샘플링.
    
    Learns the optimal drift (control force) to minimize the variance
    of the Payoff Estimator.
    페이오프 추정량의 분산을 최소화하기 위해 최적의 드리프트(제어력)를 학습합니다.
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
        가격 추정량의 분산을 최소화하도록 제어 네트워크를 학습합니다.
        
        Loss = E[ (Payoff * LikelihoodRatio)^2 ] (Second Moment / 2차 모멘트)
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
