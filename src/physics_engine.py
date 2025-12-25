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

    def simulate(self, S0, v0, T, dt, num_paths, model_type='heston'):
        n_steps = int(T / dt)
        S = torch.zeros((num_paths, n_steps + 1), device=self.device)
        v = torch.zeros((num_paths, n_steps + 1), device=self.device)
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        dt_tensor = torch.tensor(dt, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        # Drift Correction (Compensator)
        # SVJJ도 Bates와 동일하게 점프 평균(E[k])만큼 drift를 보정해줘야 함
        # SVJJ의 경우 변동성 점프는 주가 drift에 직접 영향 X (독립 가정 시 편의상 rho_J=0)
        drift_correction = 0.0
        if model_type in ['bates', 'svjj']:
            k = torch.exp(torch.tensor(self.jump_mean + 0.5 * self.jump_std**2)) - 1
            drift_correction = self.jump_lambda * k * dt

        for t in range(1, n_steps + 1):
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)
            
            dw_s = z1
            dw_v = self.rho * z1 + torch.sqrt(1 - torch.tensor(self.rho**2)) * z2

            # 1. Variance Process (Diffusion Part)
            v_prev = v[:, t-1]
            dv_diff = self.kappa * (self.theta - v_prev) * dt + \
                      self.xi * torch.sqrt(torch.clamp(v_prev, min=0)) * sqrt_dt * dw_v
            
            # 2. Asset Process (Diffusion Part)
            S_prev = S[:, t-1]
            dS_diffusion = (self.mu * dt - drift_correction) * S_prev + \
                           torch.sqrt(torch.clamp(v_prev, min=0)) * S_prev * sqrt_dt * dw_s
            
            # 3. Jump Process (Bates & SVJJ)
            jump_S = torch.zeros(num_paths, device=self.device)
            jump_v = torch.zeros(num_paths, device=self.device)
            
            if model_type in ['bates', 'svjj']:
                prob_jump = self.jump_lambda * dt
                is_jump = torch.bernoulli(torch.full((num_paths,), prob_jump, device=self.device))
                
                if torch.sum(is_jump) > 0:
                    # 주가 점프 크기 J ~ N(mean, std)
                    log_jump_size = torch.normal(self.jump_mean, self.jump_std, size=(num_paths,), device=self.device)
                    # 주가 변화: S_new = S_old * exp(J)
                    jump_impact_S = (torch.exp(log_jump_size) - 1) * S_prev
                    jump_S = jump_impact_S * is_jump
                    
                    # [SVJJ] 변동성 점프 (주가 점프와 동시 발생)
                    # Z_v ~ Exp(1/mu_v) -> Exponential Distribution
                    if model_type == 'svjj':
                        # torch.exponential은 rate(lambda) = 1/mean을 입력받음
                        # mean = vol_jump_mean -> rate = 1/vol_jump_mean
                        rate = 1.0 / (self.vol_jump_mean + 1e-6) # safety
                        vol_jump_size = torch.distributions.Exponential(rate).sample((num_paths,)).to(self.device)
                        jump_v = vol_jump_size * is_jump

            # Update State
            v[:, t] = torch.clamp(v_prev + dv_diff + jump_v, min=1e-4)
            S[:, t] = S_prev + dS_diffusion + jump_S

        return S, v