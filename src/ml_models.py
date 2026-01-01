"""
ML Models Engine / 머신러닝 모델 엔진
====================================

이 모듈은 옵션 가격 결정을 위한 전통적 머신러닝 모델들을 구현합니다.
This module implements traditional machine learning models for option pricing.

포함 모델 / Included Models:
1. Black-Scholes (Closed-form Baseline)
2. XGBoost (Gradient Boosting)
3. LSTM (Recurrent Neural Network)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


# =============================================================================
# 1. Black-Scholes Model / 블랙-숄즈 모델
# =============================================================================
class BlackScholesModel:
    """
    Black-Scholes 옵션 가격 결정 모델 / Black-Scholes Option Pricing Model.
    
    가장 단순한 기준선(Baseline) 모델입니다.
    The simplest benchmark model for option pricing.
    """
    
    def __init__(self, sigma=0.2):
        """
        Args:
            sigma: 상수 변동성 / Constant volatility (to be calibrated)
        """
        self.sigma = sigma
    
    def price(self, S0, K, T, r):
        """
        콜옵션 가격 계산 / Calculate call option price.
        
        Args:
            S0: 현재 주가 / Current price
            K: 행사가 / Strike price
            T: 만기 / Time to maturity
            r: 무위험 이자율 / Risk-free rate
        
        Returns:
            call_price: 콜옵션 가격 / Call option price
        """
        if T <= 0:
            return max(S0 - K, 0)
        
        sigma = self.sigma
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def calibrate(self, market_prices, S0, strikes, T, r):
        """
        시장 데이터에 맞춰 변동성(sigma) 캘리브레이션.
        Calibrate sigma to market data.
        """
        from scipy.optimize import minimize_scalar
        
        def loss(sigma):
            self.sigma = sigma
            model_prices = np.array([self.price(S0, K, T, r) for K in strikes])
            return np.mean((model_prices - market_prices) ** 2)
        
        result = minimize_scalar(loss, bounds=(0.01, 1.0), method='bounded')
        self.sigma = result.x
        return result.x


# =============================================================================
# 2. XGBoost Model / XGBoost 모델
# =============================================================================
class XGBoostOptionModel:
    """
    XGBoost 기반 옵션 가격 예측 모델 / XGBoost-based Option Pricing Model.
    
    입력: (Moneyness, Time-to-Maturity, Risk-free Rate)
    출력: Option Price
    """
    
    def __init__(self):
        self.model = None
    
    def train(self, X, y):
        """
        모델 학습 / Train model.
        
        Args:
            X: 특성 배열 (moneyness, T, r) / Feature array
            y: 옵션 가격 배열 / Option prices
        """
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X, y)
        except ImportError:
            # XGBoost가 없으면 sklearn의 GradientBoosting 사용
            # Fallback to sklearn if xgboost not installed
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X, y)
    
    def predict(self, X):
        """
        옵션 가격 예측 / Predict option prices.
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. / Model not trained.")
        return self.model.predict(X)


# =============================================================================
# 3. LSTM Model / LSTM 모델
# =============================================================================
class LSTMOptionModel(nn.Module):
    """
    LSTM 기반 옵션 가격 예측 모델 / LSTM-based Option Pricing Model.
    
    시계열 관점에서 옵션 가격 구조를 학습합니다.
    Learns option price structure from a time-series perspective.
    """
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, device='cuda'):
        """
        Args:
            input_dim: 입력 차원 (moneyness, T, r) / Input dimension
            hidden_dim: LSTM 은닉 차원 / LSTM hidden dimension
            num_layers: LSTM 층 수 / Number of LSTM layers
            device: 연산 장치 / Computation device
        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 레이어 / LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 출력 레이어 / Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.to(device)
    
    def forward(self, x):
        """
        순전파 / Forward pass.
        
        Args:
            x: 입력 텐서 (batch, seq_len, input_dim) / Input tensor
        
        Returns:
            output: 옵션 가격 예측 (batch,) / Option price predictions
        """
        # LSTM 순전파 / LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 출력 사용 / Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        
        # 가격 예측 / Price prediction
        output = self.fc(last_hidden)
        return output.squeeze(-1)
    
    def train_model(self, X, y, epochs=50, lr=0.001):
        """
        모델 학습 / Train model.
        
        Args:
            X: 특성 배열 (N, 3) / Feature array
            y: 타겟 배열 (N,) / Target array
            epochs: 학습 에폭 / Training epochs
            lr: 학습률 / Learning rate
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 데이터를 시퀀스로 변환 (seq_len=1) / Convert to sequence
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')
    
    def predict(self, X):
        """
        옵션 가격 예측 / Predict option prices.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1)
            predictions = self.forward(X_tensor)
        return predictions.cpu().numpy()
