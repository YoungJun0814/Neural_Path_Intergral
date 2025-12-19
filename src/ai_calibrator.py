import torch
import torch.nn as nn

class NeuralCalibrator(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        """
        input_dim: 시장 관측 데이터 (예: 현재 주가, 옵션 행사가, 만기, 현재 VIX 등)
        """
        super(NeuralCalibrator, self).__init__()
        
        # 간단한 MLP 구조 (입력 -> 은닉층 -> 파라미터 예측)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # 출력: kappa, theta, xi, rho
        )
        
        # 파라미터가 물리적으로 말이 되는 범위에 있도록 강제 (Softplus, Tanh)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Input: Market Features
        Output: Heston Parameters (kappa, theta, xi, rho)
        """
        out = self.net(x)
        
        # 각 파라미터의 특성에 맞게 활성화 함수 적용
        kappa = self.softplus(out[:, 0])        # 양수여야 함
        theta = self.softplus(out[:, 1])        # 양수여야 함
        xi = self.softplus(out[:, 2])           # 양수여야 함
        rho = -0.9 + 1.8 * torch.sigmoid(out[:, 3]) # -0.9 ~ 0.9 사이로 제한
        
        return kappa, theta, xi, rho