import torch

class PathIntegralSolver:
    def __init__(self, simulator):
        """
        simulator: src.physics_engine.HestonSimulator 인스턴스
        """
        self.sim = simulator
        self.device = simulator.device

    def compute_action(self, S, v, dt):
        """
        각 경로(Path)의 작용(Action)을 계산합니다.
        Action이 작을수록 물리적으로 발생 확률이 높은 경로입니다.
        
        Onsager-Machlup Functional을 이산화하여 계산합니다.
        """
        # 1. 미분값 근사 (Gradients)
        dS = S[:, 1:] - S[:, :-1]
        dv = v[:, 1:] - v[:, :-1]
        
        S_t = S[:, :-1]
        v_t = v[:, :-1]
        
        # 2. 드리프트(Drift) 제거 (순수 변동분 추출)
        # 주가 드리프트 제거
        drift_S = self.sim.mu * S_t * dt
        residual_S = dS - drift_S
        
        # 변동성 드리프트 제거
        drift_v = self.sim.kappa * (self.sim.theta - v_t) * dt
        residual_v = dv - drift_v
        
        # 3. 확산 행렬(Diffusion Matrix)의 역행렬 계산 (Precision Matrix)
        # Heston 모델의 노이즈는 상관관계(rho)가 있으므로 이를 분리해야 합니다.
        rho = self.sim.rho
        xi = self.sim.xi
        one_minus_rho2 = 1 - rho**2
        
        # 주가의 변동성 항 (sigma_S)
        sigma_S = torch.sqrt(v_t) * S_t
        # 변동성의 변동성 항 (sigma_v)
        sigma_v = xi * torch.sqrt(v_t)
        
        # 4. 액션(Action) 수식 (이산화된 라그랑지안)
        # L = 0.5 * (res_S^2/sig_S^2 - 2*rho*res_S*res_v/(sig_S*sig_v) + res_v^2/sig_v^2) / (1-rho^2) * dt
        
        term1 = (residual_S / (sigma_S + 1e-8)) ** 2
        term2 = -2 * rho * (residual_S * residual_v) / ((sigma_S * sigma_v) + 1e-8)
        term3 = (residual_v / (sigma_v + 1e-8)) ** 2
        
        lagrangian = 0.5 * (term1 + term2 + term3) / (one_minus_rho2 * dt)
        
        # 전체 경로에 대해 적분(합산) -> Action
        action = torch.sum(lagrangian, dim=1)
        
        return action

    def reweight_paths(self, action):
        """
        Action을 기반으로 각 경로의 확률 가중치를 계산합니다.
        Probability ~ exp(-Action)
        """
        # 수치 안정성을 위해 최소 Action을 뺍니다 (Log-Sum-Exp Trick)
        min_action = torch.min(action)
        weights = torch.exp(-(action - min_action))
        
        # 정규화 (모든 가중치의 합이 1이 되도록)
        normalized_weights = weights / torch.sum(weights)
        return normalized_weights
    
    def price_option(self, S0, K, T, r, num_paths=5000, dt=0.01):
        """
        경로 적분을 활용한 옵션 가격 결정 / Option pricing using Path Integral.
        
        Path Integral 공식: Price = E[e^{-rT} * Payoff * Weight(Action)]
        
        Args:
            S0: 초기 주가 / Initial price
            K: 행사가 / Strike price
            T: 만기 / Time to maturity
            r: 무위험 이자율 / Risk-free rate
            num_paths: 시뮬레이션 경로 수 / Number of paths
            dt: 시간 간격 / Time step
        
        Returns:
            price: 경로적분 기반 옵션 가격 / Path integral option price
        """
        # 1. 경로 생성 / Generate paths
        S_paths, v_paths = self.sim.simulate(
            S0=S0, v0=self.sim.theta, T=T, dt=dt, num_paths=num_paths,
            model_type='heston'
        )
        
        # 2. 각 경로의 Action 계산 / Compute action for each path
        action = self.compute_action(S_paths, v_paths, dt)
        
        # 3. 경로 가중치 계산 / Compute path weights
        weights = self.reweight_paths(action)
        
        # 4. 페이오프 계산 / Compute payoffs
        S_final = S_paths[:, -1]
        payoffs = torch.maximum(S_final - K, torch.tensor(0.0, device=self.device))
        
        # 5. 가중 평균 가격 (경로 적분) / Weighted average (Path Integral)
        # Price = exp(-rT) * sum(Weight * Payoff)
        discount = torch.exp(torch.tensor(-r * T, device=self.device))
        price = discount * torch.sum(weights * payoffs)
        
        return price.cpu().item()