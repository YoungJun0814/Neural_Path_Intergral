import torch

class HestonSimulator:
    def __init__(self, mu, kappa, theta, xi, rho, device='cuda'):
        self.mu = mu        # 자산 수익률
        self.kappa = kappa  # 평균 회귀 속도
        self.theta = theta  # 장기 평균 변동성
        self.xi = xi       # 변동성의 변동성 (Vol of Vol)
        self.rho = rho      # 주가-변동성 상관관계
        self.device = device

    def simulate(self, S0, v0, T, dt, num_paths):
        n_steps = int(T / dt)
        
        # 텐서 초기화 (RTX 5070 GPU 메모리에 할당)
        S = torch.zeros((num_paths, n_steps), device=self.device)
        v = torch.zeros((num_paths, n_steps), device=self.device)
        
        S[:, 0] = S0
        v[:, 0] = v0

        # 상관관계가 있는 브라운 운동 생성 (Cholesky Decomposition)
        for t in range(1, n_steps):
            # 두 개의 독립적인 랜덤 변수 생성
            z1 = torch.randn(num_paths, device=self.device)
            z2 = torch.randn(num_paths, device=self.device)
            
            # 상관관계 주입
            dw_s = z1
            dw_v = self.rho * z1 + torch.sqrt(torch.tensor(1 - self.rho**2)) * z2

            # 변동성 업데이트 (Euler-Maruyama, 0 이하 방지를 위해 ReLU 적용)
            v_prev = v[:, t-1]
            dv = self.kappa * (self.theta - v_prev) * dt + \
                 self.xi * torch.sqrt(torch.clamp(v_prev, min=0)) * torch.sqrt(torch.tensor(dt)) * dw_v
            v[:, t] = torch.clamp(v_prev + dv, min=1e-4)

            # 주가 업데이트
            S_prev = S[:, t-1]
            dS = self.mu * S_prev * dt + \
                 torch.sqrt(v[:, t]) * S_prev * torch.sqrt(torch.tensor(dt)) * dw_s
            S[:, t] = S_prev + dS

        return S, v