import torch
import torch.nn as nn

class NeuralCalibrator(nn.Module):
    """
    Neural Network for Model Calibration / 모델 캘리브레이션을 위한 신경망.
    
    Predicts model parameters (kappa, theta, xi, rho) from market observables.
    시장 관측 데이터로부터 모델 파라미터(kappa, theta, xi, rho)를 예측합니다.
    """
    def __init__(self, input_dim=5, hidden_dim=64):
        """
        Args:
            input_dim: Market observables (e.g., Price, Strike, T, VIX) / 시장 관측 데이터
            hidden_dim: Hidden layer dimension / 은닉층 차원
        """
        super(NeuralCalibrator, self).__init__()
        
        # Simple MLP structure (Input -> Hidden -> Parameter Prediction)
        # 간단한 MLP 구조 (입력 -> 은닉층 -> 파라미터 예측)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # Output: kappa, theta, xi, rho / 출력: 파라미터 4개
        )
        
        # Enforce physical constraints on parameters (Softplus, Tanh)
        # 파라미터가 물리적으로 말이 되는 범위에 있도록 강제 (Softplus, Tanh)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass / 순전파.
        
        Args:
            x: Market Features / 시장 특성
        
        Returns:
            kappa, theta, xi, rho: Predicted Heston Parameters / 예측된 Heston 파라미터
        """
        out = self.net(x)
        
        # Apply activation functions based on parameter properties
        # 각 파라미터의 특성에 맞게 활성화 함수 적용
        kappa = self.softplus(out[:, 0])        # Must be positive / 양수여야 함
        theta = self.softplus(out[:, 1])        # Must be positive / 양수여야 함
        xi = self.softplus(out[:, 2])           # Must be positive / 양수여야 함
        rho = -0.9 + 1.8 * torch.sigmoid(out[:, 3]) # Constrain to [-0.9, 0.9] / -0.9 ~ 0.9 사이로 제한
        
        return kappa, theta, xi, rho