import torch

class MarketSimulator:
    def __init__(self, mu, kappa, theta, xi, rho, 
                 jump_lambda=0.0, jump_mean=0.0, jump_std=0.0, 
                 vol_jump_mean=0.0, # SVJJ 추가 파라미터 (변동성 점프 평균)
                 device='cuda'):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        
        # Bates & SVJJ Model Parameters
        self.jump_lambda = jump_lambda
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.vol_jump_mean = vol_jump_mean # Exponential dist param for vol jump
        self.device = device

    def simulate(self, S0, v0, T, dt, num_paths, model_type='heston', override_params=None):
        # -------------------------------------------------------------------------
        # 0. Parameter Setup (Allow overrides for calibration)
        # -------------------------------------------------------------------------
        mu = self.mu
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = self.rho
        jump_lambda = self.jump_lambda
        jump_mean = self.jump_mean
        jump_std = self.jump_std
        vol_jump_mean = self.vol_jump_mean

        if override_params is not None:
            mu = override_params.get('mu', mu)
            kappa = override_params.get('kappa', kappa)
            theta = override_params.get('theta', theta)
            xi = override_params.get('xi', xi)
            rho = override_params.get('rho', rho)
            jump_lambda = override_params.get('jump_lambda', jump_lambda)
            jump_mean = override_params.get('jump_mean', jump_mean)
            jump_std = override_params.get('jump_std', jump_std)
            vol_jump_mean = override_params.get('vol_jump_mean', vol_jump_mean)

        # -------------------------------------------------------------------------
        # 1. Initialization
        # -------------------------------------------------------------------------
        n_steps = int(T / dt)
        S = torch.zeros((num_paths, n_steps + 1), device=self.device)
        v = torch.zeros((num_paths, n_steps + 1), device=self.device)
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        dt_tensor = torch.tensor(dt, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        # Drift Correction (compensator for jumps)
        drift_correction = 0.0
        if model_type in ['bates', 'svjj']:
            # k = E[e^J - 1]
            k = torch.exp(torch.tensor(jump_mean + 0.5 * jump_std**2)) - 1
            drift_correction = jump_lambda * k * dt

        # Pre-compute Rho correlations
        rho_tensor = torch.tensor(rho, device=self.device)
        sqrt_one_minus_rho2 = torch.sqrt(1 - rho_tensor**2)

        # -------------------------------------------------------------------------
        # 2. Time Stepping
        # -------------------------------------------------------------------------
        for t in range(1, n_steps + 1):
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)
            
            dw_s = z1
            dw_v = rho_tensor * z1 + sqrt_one_minus_rho2 * z2

            v_prev = v[:, t-1]
            S_prev = S[:, t-1]

            # --- Correlation & Variance Process ---
            # Heston Variance: dv = kappa(theta - v)dt + xi*sqrt(v)*dW_v
            v_val_relu = torch.clamp(v_prev, min=1e-8) # Ensure positive for sqrt
            dv_diff = kappa * (theta - v_prev) * dt + \
                      xi * torch.sqrt(v_val_relu) * sqrt_dt * dw_v
            
            # --- Asset Process ---
            # dS = (mu - drift_corr)S dt + sqrt(v)S dW_s
            dS_diffusion = (mu * dt - drift_correction) * S_prev + \
                           torch.sqrt(v_val_relu) * S_prev * sqrt_dt * dw_s
            
            # --- Jump Process ---
            jump_S = 0.0
            jump_v = 0.0
            
            if model_type in ['bates', 'svjj']:
                # Possibility of jump in this time step
                # Approximation: Bernoulli(lambda * dt)
                prob_jump = jump_lambda * dt
                is_jump = torch.rand(num_paths, device=self.device) < prob_jump
                
                if is_jump.any():
                    # 1. Price Jump
                    log_jump_size = torch.normal(jump_mean, jump_std, size=(num_paths,), device=self.device)
                    # Yields factor (e^J - 1)
                    jump_factor = (torch.exp(log_jump_size) - 1)
                    jump_S = jump_factor * S_prev * is_jump
                    
                    # 2. Volatility Jump (SVJJ only)
                    if model_type == 'svjj':
                        # Exponential distribution with mean = vol_jump_mean
                        # Rate = 1 / mean
                        rate = 1.0 / (vol_jump_mean + 1e-6)
                        vol_jump_size = torch.distributions.Exponential(rate).sample((num_paths,)).to(self.device)
                        jump_v = vol_jump_size * is_jump

            # Update & Clamp
            v_new = v_prev + dv_diff + jump_v
            v[:, t] = torch.clamp(v_new, min=1e-8) # Physics constraint: vol > 0
            
            S_new = S_prev + dS_diffusion + jump_S
            S[:, t] = torch.clamp(S_new, min=1e-8) # Physics constraint: price > 0

        return S, v

    def simulate_controlled(self, S0, v0, T, dt, num_paths, model_type='heston', 
                          control_fn=None, barrier_level=None, barrier_type='down-out'):
        """
        Controlled simulation for Neural Path Integral and Barrier Options.
        Uses list accumulation to avoid in-place operations (autograd compatible).
        """
        mu = self.mu
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = self.rho
        device = self.device
        
        n_steps = max(1, int(T / dt))
        
        # Use lists to avoid in-place ops
        S_list = []
        v_list = []
        
        # Initial values
        curr_S = torch.ones(num_paths, device=device) * S0
        curr_v = torch.ones(num_paths, device=device) * v0
        
        S_list.append(curr_S)
        v_list.append(curr_v)
        
        # Girsanov Integrals (accumulated scalars)
        int_u_dW = torch.zeros(num_paths, device=device)
        int_u_sq_dt = torch.zeros(num_paths, device=device)
        
        # Barrier Monitoring
        barrier_hit = torch.zeros(num_paths, dtype=torch.bool, device=device)
        
        # Running Integral for Asian Option
        running_int_S = torch.zeros(num_paths, device=device)
        
        # Pre-compute constants
        dt_tensor = torch.tensor(dt, device=device)
        sqrt_dt = torch.sqrt(dt_tensor)
        rho_tensor = torch.tensor(rho, device=device)
        sqrt_one_minus_rho2 = torch.sqrt(1 - rho_tensor**2)
        
        for t_idx in range(1, n_steps + 1):
            t_curr = (t_idx - 1) * dt
            
            z1 = torch.randn(num_paths, device=device)
            z2 = torch.randn(num_paths, device=device)
            
            # ... (Same noise gen) ...
            dw_s = z1
            dw_v = rho_tensor * z1 + sqrt_one_minus_rho2 * z2
            
            # 1. Variance Process
            v_val_relu = torch.clamp(curr_v, min=1e-8)
            dv = kappa * (theta - curr_v) * dt + xi * torch.sqrt(v_val_relu) * sqrt_dt * dw_v
            next_v = torch.clamp(curr_v + dv, min=1e-8)
            
            # 2. Control Force (with Running Average)
            if control_fn is not None:
                # Calculate current running average A_t = (1/t) * int_S
                # For stability, if t=0, use curr_S
                if t_curr > 1e-6:
                    curr_avg = running_int_S / t_curr
                else:
                    curr_avg = curr_S
                
                # Input to control_fn: (t, S, v, A)
                u_t = control_fn(t_curr, curr_S, curr_v, curr_avg)
            else:
                u_t = torch.zeros(num_paths, device=device)
                
            # 3. Asset Process
            sigma_s = torch.sqrt(v_val_relu)
            drift_term = (mu + sigma_s * u_t) * curr_S * dt
            diffusion_term = sigma_s * curr_S * sqrt_dt * dw_s
            dS = drift_term + diffusion_term
            next_S = torch.clamp(curr_S + dS, min=1e-8)
            
            # Update Running Integral (Trapezoidal rule or simple left-point)
            # Using left-point for simplicity: int += S_t * dt
            running_int_S = running_int_S + curr_S * dt
            
            # Append to lists
            S_list.append(next_S)
            v_list.append(next_v)
            
            # 4. Girsanov Update
            int_u_dW = int_u_dW + u_t * z1 * sqrt_dt
            int_u_sq_dt = int_u_sq_dt + (u_t ** 2) * dt
            
            # 5. Barrier Check
            if barrier_level is not None:
                if barrier_type == 'down-out':
                    hit_mask = next_S <= barrier_level
                    barrier_hit = barrier_hit | hit_mask
                elif barrier_type == 'up-out':
                    hit_mask = next_S >= barrier_level
                    barrier_hit = barrier_hit | hit_mask
            
            # Update state
            curr_S = next_S
            curr_v = next_v

        # Stack lists
        S = torch.stack(S_list, dim=1)  # (num_paths, n_steps+1)
        v = torch.stack(v_list, dim=1)
        
        # Log Weights
        log_weights = -int_u_dW - 0.5 * int_u_sq_dt
        
        # Return running_int_S for Asian Payoff calculation
        return S, v, log_weights, barrier_hit, running_int_S
# Rough Volatility (rBergomi) Model / 거친 변동성 (rBergomi) 모델
# =============================================================================
# Reference: Bayer, Friz, Gatheral (2016) "Pricing under rough volatility"
# 참고 문헌: Bayer, Friz, Gatheral (2016) "거친 변동성 하의 옵션 가격 결정"
# =============================================================================

class FractionalBrownianMotion:
    """
    분수 브라운 운동 (fBm) 생성기 / Fractional Brownian Motion Generator.
    
    Hurst 지수 H < 0.5 인 경우 "거친(Rough)" 경로를 생성합니다.
    When H < 0.5, generates "rough" paths with fractal-like behavior.
    
    구현 방식: Cholesky 분해법 (정확하지만 O(N^2) 메모리 사용)
    Implementation: Cholesky decomposition (exact but O(N^2) memory)
    """
    
    def __init__(self, H=0.1, device='cuda'):
        """
        Args:
            H: Hurst 지수 / Hurst exponent. (0 < H < 0.5 for rough volatility)
               H = 0.5 일 때 표준 브라운 운동과 동일.
               H = 0.5 corresponds to standard Brownian motion.
            device: 연산 장치 / Computation device ('cuda' or 'cpu')
        """
        self.H = H
        self.device = device
    
    def _covariance_matrix(self, n_steps, dt):
        """
        fBm 공분산 행렬 생성 / Generate fBm covariance matrix.
        
        Cov(W_H(s), W_H(t)) = 0.5 * (|s|^(2H) + |t|^(2H) - |t-s|^(2H))
        
        Args:
            n_steps: 시간 스텝 수 / Number of time steps
            dt: 시간 간격 / Time step size
        
        Returns:
            공분산 행렬 (n_steps x n_steps) / Covariance matrix
        """
        H = self.H
        times = torch.arange(1, n_steps + 1, device=self.device, dtype=torch.float32) * dt
        
        # 행렬 구성 / Construct matrix
        t_i = times.unsqueeze(1)  # (n_steps, 1)
        t_j = times.unsqueeze(0)  # (1, n_steps)
        
        # 공분산 공식 / Covariance formula
        cov = 0.5 * (
            torch.abs(t_i) ** (2 * H) + 
            torch.abs(t_j) ** (2 * H) - 
            torch.abs(t_i - t_j) ** (2 * H)
        )
        
        # 수치 안정성을 위해 작은 값 추가 / Add small value for numerical stability
        cov = cov + 1e-8 * torch.eye(n_steps, device=self.device)
        
        return cov
    
    def generate(self, n_paths, n_steps, dt):
        """
        fBm 경로 생성 (Cholesky 분해) / Generate fBm paths using Cholesky decomposition.
        
        Args:
            n_paths: 경로 수 / Number of paths
            n_steps: 시간 스텝 수 / Number of time steps
            dt: 시간 간격 / Time step size
        
        Returns:
            W_H: fBm 경로 (n_paths, n_steps+1), 첫 번째 열은 0 / fBm paths, first column is 0
        """
        # 공분산 행렬 및 Cholesky 분해 / Covariance matrix and Cholesky decomposition
        cov = self._covariance_matrix(n_steps, dt)
        L = torch.linalg.cholesky(cov)  # 하삼각 행렬 / Lower triangular matrix
        
        # 표준 정규 난수 생성 / Generate standard normal random variables
        Z = torch.randn(n_paths, n_steps, device=self.device)
        
        # fBm 값 계산: W_H = L @ Z^T -> 전치해서 (n_paths, n_steps)
        # Compute fBm: W_H = L @ Z^T -> transpose to (n_paths, n_steps)
        W_H = (L @ Z.T).T
        
        # 시작점 0 추가 / Prepend zero (W_H(0) = 0)
        zeros = torch.zeros(n_paths, 1, device=self.device)
        W_H = torch.cat([zeros, W_H], dim=1)
        
        return W_H


class RBergomiSimulator:
    """
    rBergomi (Rough Bergomi) 변동성 모델 시뮬레이터 / rBergomi Volatility Model Simulator.
    
    변동성 동역학 / Volatility Dynamics:
        v_t = xi * exp(eta * W_H(t) - 0.5 * eta^2 * t^(2H))
    
    가격 동역학 / Price Dynamics:
        dS_t = sqrt(v_t) * S_t * dW_t
    
    여기서 W_H는 Hurst 지수 H인 분수 브라운 운동.
    Where W_H is fractional Brownian motion with Hurst exponent H.
    """
    
    def __init__(self, H=0.1, eta=1.9, xi=0.235, rho=-0.9, device='cuda'):
        """
        Args:
            H: Hurst 지수 / Hurst exponent (0 < H < 0.5)
            eta: 변동성의 변동성 / Volatility of volatility
            xi: 포워드 분산 초기값 / Forward variance initial level
            rho: 상관계수 / Correlation between price and volatility
            device: 연산 장치 / Computation device
        """
        self.H = H
        self.eta = eta
        self.xi = xi
        self.rho = rho
        self.device = device
        self.fBm = FractionalBrownianMotion(H=H, device=device)
    
    def simulate(self, S0, T, dt, num_paths, mu=0.0, override_params=None):
        """
        rBergomi 모델로 가격 및 변동성 경로 시뮬레이션.
        Simulate price and volatility paths using rBergomi model.
        
        Args:
            S0: 초기 주가 / Initial asset price
            T: 만기 시간 / Time to maturity
            dt: 시간 간격 / Time step size
            num_paths: 경로 수 / Number of paths
            mu: 드리프트 (리스크 중립 시 0) / Drift (0 for risk-neutral)
            override_params: 파라미터 오버라이드 / Parameter overrides for calibration
        
        Returns:
            S: 가격 경로 (num_paths, n_steps+1) / Price paths
            v: 변동성 경로 (num_paths, n_steps+1) / Volatility paths
        """
        # 파라미터 오버라이드 처리 / Handle parameter overrides
        H = self.H
        eta = self.eta
        xi = self.xi
        rho = self.rho
        
        if override_params is not None:
            H = override_params.get('H', H)
            eta = override_params.get('eta', eta)
            xi = override_params.get('xi', xi)
            rho = override_params.get('rho', rho)
            # fBm 재생성이 필요할 수 있음 / May need to regenerate fBm
            if 'H' in override_params:
                self.fBm = FractionalBrownianMotion(H=H, device=self.device)
        
        n_steps = int(T / dt)
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        # -------------------------------------------------------------------------
        # 1. fBm 경로 생성 / Generate fBm paths
        # -------------------------------------------------------------------------
        W_H = self.fBm.generate(num_paths, n_steps, dt)  # (num_paths, n_steps+1)
        
        # -------------------------------------------------------------------------
        # 2. 상관된 브라운 운동 생성 / Generate correlated Brownian motion for price
        # -------------------------------------------------------------------------
        # dW_S = rho * dW_H_increment + sqrt(1-rho^2) * dZ
        # 여기서 dW_H_increment ≈ W_H[t] - W_H[t-1] (근사)
        Z = torch.randn(num_paths, n_steps, device=self.device)
        
        # fBm 증분 계산 / Compute fBm increments
        dW_H = W_H[:, 1:] - W_H[:, :-1]  # (num_paths, n_steps)
        
        # 상관된 브라운 운동 / Correlated Brownian motion
        sqrt_one_minus_rho2 = torch.sqrt(torch.tensor(1.0 - rho**2, device=self.device))
        dW_S = rho * dW_H + sqrt_one_minus_rho2 * Z * sqrt_dt
        
        # -------------------------------------------------------------------------
        # 3. 변동성 경로 계산 / Compute volatility paths
        # -------------------------------------------------------------------------
        # v_t = xi * exp(eta * W_H(t) - 0.5 * eta^2 * t^(2H))
        times = torch.arange(n_steps + 1, device=self.device, dtype=torch.float32) * dt
        times = times.unsqueeze(0)  # (1, n_steps+1)
        
        # 변동성 계산 / Volatility computation
        v = xi * torch.exp(eta * W_H - 0.5 * eta**2 * times ** (2 * H))
        v = torch.clamp(v, min=1e-8)  # 음수 방지 / Prevent negative values
        
        # -------------------------------------------------------------------------
        # 4. 가격 경로 계산 / Compute price paths
        # -------------------------------------------------------------------------
        S = torch.zeros(num_paths, n_steps + 1, device=self.device)
        S[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            v_prev = v[:, t - 1]
            S_prev = S[:, t - 1]
            
            # dS = mu * S * dt + sqrt(v) * S * dW_S
            dS = mu * S_prev * dt + torch.sqrt(v_prev) * S_prev * dW_S[:, t - 1]
            S[:, t] = torch.clamp(S_prev + dS, min=1e-8)
        
        return S, v