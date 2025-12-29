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