"""
Differential Privacy Utilities for DP-SGD Training
Based on Opacus library but adapted for DeepUWF architecture
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np
from opacus.accountants import create_accountant, IAccountant, RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from functools import partial


class PrivacyBudget:
    """Privacy budget configuration and accounting"""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_per_sample_grad_norm: float = 1.0,
        mechanism: str = "rdp",
        sigma: Optional[float] = None,
    ):
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_per_sample_grad_norm = float(max_per_sample_grad_norm)
        self.mechanism = str(mechanism)

        if sigma is None or sigma == "None" or sigma == "null":
            self.sigma = None
        else:
            self.sigma = float(sigma)

        self.accountant = create_accountant(mechanism=self.mechanism)
        self.steps = 0


    def compute_sigma(
        self,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        eps_tol: float = 1e-6,
    ) -> float:
        target_epsilon = float(target_epsilon)
        target_delta = float(target_delta)
        sample_rate = float(sample_rate)
        steps = int(steps)
        eps_tol = float(eps_tol)

        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=self.mechanism,
            epsilon_tolerance=eps_tol,
        )

        self.sigma = float(sigma)
        return self.sigma

    
    def compute_final_epsilon(
        self,
        steps: int,
        sample_rate: float,
        alphas: Optional[List[float]] = None,
    ) -> float:
        """
        Compute final epsilon after training
        
        Args:
            steps: 实际训练步数
            sample_rate: 采样率
            alphas: RDP 计算用的 alpha 值
            
        Returns:
            final_epsilon: 最终隐私预算
        """
        if self.sigma is None:
            raise ValueError("Sigma must be set before computing final epsilon")
        
        self.accountant.history = [(self.sigma, sample_rate, steps)]
        kwargs = {}
        if isinstance(self.accountant, RDPAccountant):
            default_alphas = RDPAccountant.DEFAULT_ALPHAS
            if alphas:
                default_alphas = default_alphas + alphas
            kwargs["alphas"] = default_alphas
        
        final_epsilon = self.accountant.get_epsilon(delta=self.delta, **kwargs)
        return final_epsilon


class GradientClipper:
    """
    Clips gradients by per-sample L2 norm
    """
    
    def __init__(self, max_per_sample_grad_norm: float):
        """
        Args:
            max_per_sample_grad_norm: 每个样本梯度的L2范数上限
        """
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        
    def __call__(
        self,
        grads: List[torch.Tensor],
        model: nn.Module,
        batch_size: int,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Clip gradients by per-sample L2 norm
        
        Args:
            grads: 梯度列表
            model: 模型
            batch_size: 批大小
            
        Returns:
            clipped_grads: 剪切后的梯度
            clipping_factors: 每个样本的剪切因子
        """
        device = grads[0].device if grads else torch.device("cpu")
        clipped_grads = []
        clipping_factors = []
        
        # 计算每个样本的梯度范数
        for grad in grads:
            if grad.dim() > 1:
                # 重新reshape为 (batch_size, -1)
                grad_reshaped = grad.reshape(batch_size, -1)
                grad_norms = torch.norm(grad_reshaped, p=2, dim=1)
            else:
                grad_norms = torch.full((batch_size,), 0.0, device=device)
                
            # 计算剪切因子
            clipping_factor = torch.clamp(
                self.max_per_sample_grad_norm / (grad_norms + 1e-12),
                max=1.0
            )
            clipping_factors.append(clipping_factor.detach().cpu().numpy())
            
            # 应用剪切
            if grad.dim() > 1:
                clipped_grad = grad * clipping_factor.view(-1, *([1] * (grad.dim() - 1)))
            else:
                clipped_grad = grad * clipping_factor.view(-1)
            clipped_grads.append(clipped_grad)
            
        return clipped_grads, clipping_factors


class DPNoiseAdder:
    """
    Add Gaussian noise for Differential Privacy
    """
    
    def __init__(self, sigma: float, max_per_sample_grad_norm: float, device: torch.device):
        """
        Args:
            sigma: 噪声乘数
            max_per_sample_grad_norm: 每个样本梯度范数上限
            device: torch device
        """
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.device = device
        
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to gradient
        
        Args:
            grad: 要添加噪声的梯度
            
        Returns:
            noisy_grad: 添加噪声后的梯度
        """
        noise_scale = self.sigma * self.max_per_sample_grad_norm
        noise = torch.normal(
            mean=0.0,
            std=noise_scale,
            size=grad.shape,
            device=self.device,
            dtype=grad.dtype
        )
        return grad + noise


class DPTrainer:
    """
    Trainer with DP-SGD capability
    """
    
    def __init__(
        self,
        model: nn.Module,
        privacy_budget: PrivacyBudget,
        batch_size: int,
        dataset_size: int,
        device: torch.device,
        effective_batch_size: Optional[int] = None,
    ):
        """
        Args:
            model: 模型
            privacy_budget: 隐私预算配置
            batch_size: 批大小
            dataset_size: 数据集大小
            device: torch device
        """
        self.model = model
        self.privacy_budget = privacy_budget
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.device = device
        self.effective_batch_size = int(effective_batch_size or batch_size)
        
        self.gradient_clipper = GradientClipper(privacy_budget.max_per_sample_grad_norm)
        
        self.sample_rate = batch_size / dataset_size
        self.accumulated_grads = None
        
    def compute_per_sample_grads(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
    ) -> List[torch.Tensor]:
        """
        Compute per-sample gradients
        
        Args:
            batch: (images, labels)
            criterion: 损失函数
            
        Returns:
            per_sample_grads: 每个样本的梯度
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        batch_size = images.size(0)
        
        # 计算逐样本梯度
        per_sample_grads = []
        for i in range(batch_size):
            self.model.zero_grad()
            
            # 前向传播
            outputs = self.model(images[i:i+1])
            loss = criterion(outputs, labels[i:i+1])
            
            # 反向传播
            loss.backward()
            
            # 收集梯度
            sample_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                           for p in self.model.parameters()]
            per_sample_grads.append(sample_grads)
        
        return per_sample_grads
    
    def apply_dp_gradient_update(
        self,
        per_sample_grads: List[List[torch.Tensor]],
        add_noise: bool = True,
        effective_batch_size: Optional[int] = None,
        return_metadata: bool = False,
    ):
        """
        Apply DP-SGD: global per-sample clipping, Gaussian noise, then average.
        
        Args:
            per_sample_grads: 逐样本梯度列表
            add_noise: 是否添加噪声
            effective_batch_size: 用于平均的 effective batch size
            return_metadata: 是否返回裁剪/噪声统计信息
            
        Returns:
            dp_grads: clipping/noise/average 后暴露给 optimizer 的梯度
        """
        batch_size = len(per_sample_grads)
        num_params = len(per_sample_grads[0])
        if batch_size == 0:
            raise ValueError("per_sample_grads must not be empty")

        divisor = int(effective_batch_size or self.effective_batch_size or batch_size)
        if divisor <= 0:
            raise ValueError("effective_batch_size must be positive")

        clip_norm = self.privacy_budget.max_per_sample_grad_norm
        device = per_sample_grads[0][0].device

        # RePrAAIMI/DP-SGD style: compute one global L2 norm per sample over
        # all trainable parameters, then use that single clipping factor for
        # every parameter tensor of the sample.
        grad_norms = []
        for sample_idx in range(batch_size):
            sq_norm = torch.zeros((), device=device)
            for grad in per_sample_grads[sample_idx]:
                sq_norm = sq_norm + grad.detach().pow(2).sum()
            grad_norms.append(torch.sqrt(sq_norm))

        grad_norms = torch.stack(grad_norms)
        clipping_factors = torch.clamp(clip_norm / (grad_norms + 1e-12), max=1.0)

        sigma = self.privacy_budget.sigma
        if add_noise and sigma is None:
            raise ValueError(
                "privacy_budget.sigma is None. Call update_privacy_budget() "
                "before DP training, or set dp.sigma in the config."
            )

        noise_scale = float(sigma) * clip_norm if add_noise else 0.0
        aggregated_grads = []

        for param_idx in range(num_params):
            summed_grad = torch.zeros_like(per_sample_grads[0][param_idx])
            for sample_idx in range(batch_size):
                summed_grad = (
                    summed_grad
                    + per_sample_grads[sample_idx][param_idx] * clipping_factors[sample_idx]
                )

            if add_noise:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=summed_grad.shape,
                    device=summed_grad.device,
                    dtype=summed_grad.dtype,
                )
                summed_grad = summed_grad + noise

            aggregated_grads.append(summed_grad / divisor)

        if not return_metadata:
            return aggregated_grads

        metadata = {
            "batch_size": int(batch_size),
            "effective_batch_size": int(divisor),
            "clip_norm": float(clip_norm),
            "sigma": float(sigma) if sigma is not None else None,
            "noise_scale": float(noise_scale),
            "num_clipped": int((clipping_factors < 1.0).sum().item()),
            "clip_fraction": float((clipping_factors < 1.0).float().mean().item()),
            "grad_norm_min": float(grad_norms.min().item()),
            "grad_norm_mean": float(grad_norms.mean().item()),
            "grad_norm_max": float(grad_norms.max().item()),
        }
        return aggregated_grads, metadata


def update_privacy_budget(
    privacy_budget: PrivacyBudget,
    num_epochs: int,
    num_batches_per_epoch: int,
    batch_size: int,
    dataset_size: int,
    alphas: Optional[List[float]] = None,
    grad_acc_steps: int = 1,
) -> float:
    """
    Compute and update privacy budget for training
    
    Args:
        privacy_budget: 隐私预算对象
        num_epochs: 训练轮数
        num_batches_per_epoch: 每个epoch的batch数
        batch_size: 批大小
        dataset_size: 数据集大小
        alphas: RDP alpha 值
        
    Returns:
        final_epsilon: 最终隐私预算
    """
    grad_acc_steps = int(grad_acc_steps or 1)
    if grad_acc_steps <= 0:
        raise ValueError("grad_acc_steps must be positive")

    effective_batch_size = int(batch_size) * grad_acc_steps
    sample_rate = effective_batch_size / dataset_size
    steps_per_epoch = num_batches_per_epoch // grad_acc_steps
    if steps_per_epoch <= 0:
        raise ValueError(
            "num_batches_per_epoch must be at least grad_acc_steps for DP accounting"
        )
    steps = num_epochs * steps_per_epoch
    
    # 计算所需的sigma
    sigma = privacy_budget.compute_sigma(
        target_epsilon=privacy_budget.epsilon,
        target_delta=privacy_budget.delta,
        sample_rate=sample_rate,
        steps=steps,
    )
    
    print(f"[Privacy] Computed sigma={sigma:.6f} for epsilon={privacy_budget.epsilon}, delta={privacy_budget.delta}")
    print(
        f"[Privacy] Sample rate={sample_rate:.6f}, steps={steps}, "
        f"effective_batch_size={effective_batch_size}, grad_acc_steps={grad_acc_steps}"
    )
    
    # 计算最终epsilon
    final_epsilon = privacy_budget.compute_final_epsilon(
        steps=steps,
        sample_rate=sample_rate,
        alphas=alphas,
    )
    
    print(f"[Privacy] Final epsilon after {num_epochs} epochs: {final_epsilon:.6f}")
    
    return final_epsilon
