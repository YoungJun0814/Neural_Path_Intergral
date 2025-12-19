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