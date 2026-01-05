import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# 1. Reproducibility Tools (Essential for Phase 4) / 재현성 보장 도구 (Phase 4 필수)
def set_seed(seed=42):
    """
    Fix all random seeds for reproducibility.
    모든 랜덤 시드를 고정하여 실험 결과를 재현 가능하게 만듭니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 2. Tensor Conversion Tools / 텐서 변환 도구
def to_numpy(tensor):
    """
    Safely convert PyTorch Tensor to Numpy array.
    (Moves to CPU if on GPU).
    
    PyTorch Tensor를 Numpy 배열로 안전하게 변환합니다.
    (GPU에 있어도 CPU로 가져옵니다.)
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor

# 3. Dataset Class (Used in Phase 3) / 데이터셋 클래스 (Phase 3에서 사용)
class HestonDataset(Dataset):
    def __init__(self, num_samples=1000, T=1.0, dt=1/252, device='cuda'):
        self.num_samples = num_samples
        self.T = T
        self.dt = dt
        self.device = device
        
        # Pre-generate synthetic data / 학습 데이터 미리 생성 (Synthetic Data)
        # Create random parameter combinations / 랜덤한 파라미터 조합을 만듭니다.
        self.kappas = torch.rand(num_samples, device=device) * 4.0 + 0.1
        self.thetas = torch.rand(num_samples, device=device) * 0.5 + 0.01
        self.xis = torch.rand(num_samples, device=device) * 0.9 + 0.1
        self.rhos = torch.rand(num_samples, device=device) * 1.8 - 0.9
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'kappa': self.kappas[idx],
            'theta': self.thetas[idx],
            'xi': self.xis[idx],
            'rho': self.rhos[idx]
        }

# 4. Path Visualization Tools (Essential for Phase 4) / 경로 시각화 도구 (Phase 4 필수)
def plot_weighted_paths(S_paths, weights, num_to_plot=30, title="Weighted Paths"):
    """
    Visualize top 'Classical Paths' with lowest Action (highest weight).
    Darker lines indicate physically more probable paths.
    
    Action이 가장 낮은(가중치가 높은) 상위 'Classical Path'들을 시각화합니다.
    물리적으로 가장 타당한 경로들이 진하게 표시됩니다.
    """
    # Sort by weight / 가중치가 높은 순서대로 정렬
    sorted_indices = torch.argsort(weights, descending=True)
    top_indices = sorted_indices[:num_to_plot]
    
    S_cpu = to_numpy(S_paths)
    weights_cpu = to_numpy(weights)
    
    plt.figure(figsize=(10, 6))
    
    # Draw top paths / 상위 경로들 그리기
    # Use transparency (alpha) relative to the top path's weight
    # 1등 경로의 가중치를 기준으로 투명도(alpha)를 조절합니다.
    max_weight = weights_cpu[top_indices[0]]
    
    for idx in top_indices:
        # Darker for higher weight / 가중치가 높을수록 진하게, 낮을수록 흐리게
        alpha = weights_cpu[idx] / max_weight 
        plt.plot(S_cpu[idx], color='blue', alpha=max(0.1, min(1.0, alpha)), linewidth=1)
        
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True, alpha=0.3)
    plt.show()