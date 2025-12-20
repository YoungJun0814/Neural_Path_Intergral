import torch

class MarketSimulator:
    def __init__(self, mu, kappa, theta, xi, rho, 
                 jump_lambda=0.0, jump_mean=0.0, jump_std=0.0, device='cuda'):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        
        # Bates Model Parameters
        self.jump_lambda = jump_lambda
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.device = device

    def simulate(self, S0, v0, T, dt, num_paths, model_type='heston'):
        n_steps = int(T / dt)
        S = torch.zeros((num_paths, n_steps + 1), device=self.device) # +1 for t=0
        v = torch.zeros((num_paths, n_steps + 1), device=self.device)
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        dt_tensor = torch.tensor(dt, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)

        # [논문 디테일] 점프에 의한 기대 수익률 상승을 상쇄하는 'Drift Correction' (Compensator)
        # 이걸 안 하면 Bates 모델의 주가가 Heston보다 무조건 높게 나옵니다.
        # k = E[e^J - 1] (점프로 인한 평균 수익률 변화분)
        if model_type == 'bates':
            k = torch.exp(torch.tensor(self.jump_mean + 0.5 * self.jump_std**2)) - 1
            drift_correction = self.jump_lambda * k * dt
        else:
            drift_correction = 0.0

        for t in range(1, n_steps + 1):
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)
            
            dw_s = z1
            dw_v = self.rho * z1 + torch.sqrt(1 - torch.tensor(self.rho**2)) * z2

            # 1. Variance Process (Heston)
            v_prev = v[:, t-1]
            dv = self.kappa * (self.theta - v_prev) * dt + \
                 self.xi * torch.sqrt(torch.clamp(v_prev, min=0)) * sqrt_dt * dw_v
            v[:, t] = torch.clamp(v_prev + dv, min=1e-4)

            # 2. Asset Process (Diffusion)
            S_prev = S[:, t-1]
            # 여기서 drift_correction을 빼주는 것이 Bates 모델의 정석입니다.
            dS_diffusion = (self.mu * dt - drift_correction) * S_prev + \
                           torch.sqrt(v[:, t]) * S_prev * sqrt_dt * dw_s
            
            # 3. Jump Process (Bates)
            jump_factor = torch.zeros(num_paths, device=self.device)
            if model_type == 'bates':
                prob_jump = self.jump_lambda * dt
                is_jump = torch.bernoulli(torch.full((num_paths,), prob_jump, device=self.device))
                
                if torch.sum(is_jump) > 0:
                    # 점프 크기 J ~ N(mean, std)
                    jump_size = torch.normal(self.jump_mean, self.jump_std, size=(num_paths,), device=self.device)
                    # 주가 변화: S_new = S_old * exp(J) -> 변화량 = S_old * (exp(J) - 1)
                    jump_impact = torch.exp(jump_size) - 1
                    jump_factor = S_prev * jump_impact * is_jump

            S[:, t] = S_prev + dS_diffusion + jump_factor

        return S, v