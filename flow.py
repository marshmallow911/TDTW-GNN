import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import math
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.decomposition import PCA


class MultiDimMLP(nn.Module):
    """
    数值稳定的多维MLP网络
    """

    def __init__(self, input_shape: Union[int, Tuple], hidden_dims: List[int],
                 output_shape: Union[int, Tuple], init_zeros: bool = True,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()

        # 处理输入和输出形状
        if isinstance(input_shape, int):
            self.input_dim = input_shape
            self.input_shape = (input_shape,)
        else:
            self.input_dim = np.prod(input_shape)
            self.input_shape = input_shape

        if isinstance(output_shape, int):
            self.output_dim = output_shape
            self.output_shape = (output_shape,)
        else:
            self.output_dim = np.prod(output_shape)
            self.output_shape = output_shape

        # 构建网络层
        layers = []
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 添加批归一化（除了最后一层）
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

                # 激活函数
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.1))

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        # 改进的初始化
        self._initialize_weights(init_zeros)

    def _initialize_weights(self, init_zeros: bool):
        """改进的权重初始化"""
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                if i == len(self.net) - 1 and init_zeros:
                    # 最后一层零初始化
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                else:
                    # Xavier初始化
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        # 展平输入
        x_flat = x.view(batch_size, -1)
        # 通过网络
        out = self.net(x_flat)
        # 重塑为目标形状
        return out.view(batch_size, *self.output_shape)


class StableMaskedAffineFlow(nn.Module):
    """
    数值稳定的Masked Affine Flow
    """

    def __init__(self, shape: Union[int, Tuple], mask: torch.Tensor,
                 hidden_dims: List[int] = None, init_zeros: bool = True,
                 scale_activation: str = 'sigmoid', scale_bound: float = 5.0):
        super().__init__()

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

        self.register_buffer('mask', mask)
        self.scale_activation = scale_activation
        self.scale_bound = scale_bound

        # 默认隐藏层维度
        if hidden_dims is None:
            total_dim = np.prod(self.shape)
            hidden_dims = [total_dim * 2, total_dim * 2]

        # 计算被mask的部分的维度
        self.masked_dim = int(torch.sum(mask).item())
        self.unmasked_dim = int(torch.sum(1 - mask).item())

        if self.masked_dim == 0 or self.unmasked_dim == 0:
            raise ValueError("Mask must have both 0s and 1s")

        # 尺度和平移网络
        self.scale_net = MultiDimMLP(
            self.masked_dim, hidden_dims, self.unmasked_dim,
            init_zeros=init_zeros, activation='relu', dropout=0.1
        )
        self.translate_net = MultiDimMLP(
            self.masked_dim, hidden_dims, self.unmasked_dim,
            init_zeros=init_zeros, activation='relu', dropout=0.1
        )

    def _get_scale_and_translate(self, x_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算稳定的尺度和平移参数"""
        raw_scale = self.scale_net(x_cond)
        translate = self.translate_net(x_cond)

        # 应用不同的scale激活函数以保证数值稳定性
        if self.scale_activation == 'sigmoid':
            # 使用sigmoid将scale限制在合理范围内
            scale = torch.sigmoid(raw_scale) * 2 * self.scale_bound - self.scale_bound
        elif self.scale_activation == 'tanh':
            # 使用tanh
            scale = torch.tanh(raw_scale) * self.scale_bound
        elif self.scale_activation == 'clamp':
            # 直接clamp
            scale = torch.clamp(raw_scale, -self.scale_bound, self.scale_bound)
        else:
            raise ValueError(f"Unknown scale activation: {self.scale_activation}")

        return scale, translate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，增加数值稳定性检查
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")

        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # 应用mask分离输入
        mask_expanded = self.mask.bool().unsqueeze(0).expand(batch_size, -1)
        x_masked_indices = mask_expanded
        x_unmasked_indices = ~mask_expanded

        # 提取条件输入
        x_cond = x_flat[x_masked_indices].view(batch_size, -1)
        x_unmasked = x_flat[x_unmasked_indices].view(batch_size, -1)

        # 计算尺度和平移参数
        scale, translate = self._get_scale_and_translate(x_cond)

        # 应用仿射变换
        y_unmasked = x_unmasked * torch.exp(scale) + translate

        # 重组输出
        y_flat = x_flat.clone()
        y_flat[x_unmasked_indices] = y_unmasked.flatten()

        y = y_flat.view(batch_size, *self.shape)

        # 计算log determinant
        log_det = torch.sum(scale, dim=1)

        # 检查输出是否包含NaN或Inf
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"Warning: Output contains NaN or Inf. Scale range: [{scale.min():.3f}, {scale.max():.3f}]")
            print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"Scale stats: mean={scale.mean():.3f}, std={scale.std():.3f}")

        if torch.isnan(log_det).any() or torch.isinf(log_det).any():
            print(f"Warning: Log det contains NaN or Inf. Scale range: [{scale.min():.3f}, {scale.max():.3f}]")

        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播，增加数值稳定性检查
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("Input contains NaN or Inf values")

        batch_size = y.shape[0]
        y_flat = y.view(batch_size, -1)

        # 应用mask分离输出
        mask_expanded = self.mask.bool().unsqueeze(0).expand(batch_size, -1)
        y_masked_indices = mask_expanded
        y_unmasked_indices = ~mask_expanded

        # 提取条件输入
        y_cond = y_flat[y_masked_indices].view(batch_size, -1)
        y_unmasked = y_flat[y_unmasked_indices].view(batch_size, -1)

        # 计算尺度和平移参数
        scale, translate = self._get_scale_and_translate(y_cond)

        # 应用逆仿射变换
        x_unmasked = (y_unmasked - translate) * torch.exp(-scale)

        # 重组输出
        x_flat = y_flat.clone()
        x_flat[y_unmasked_indices] = x_unmasked.flatten()

        x = x_flat.view(batch_size, *self.shape)

        # 计算log determinant (负值)
        log_det = -torch.sum(scale, dim=1)

        return x, log_det


class MultiDimRealNVP(nn.Module):
    """
    数值稳定的多维Real NVP模型
    """

    def __init__(self, shape: Union[int, Tuple], num_layers: int = 8,
                 hidden_dims: List[int] = None, mask_type: str = 'checkerboard',
                 scale_activation: str = 'sigmoid', scale_bound: float = 5.0):
        super().__init__()

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

        self.num_layers = num_layers
        self.flows = nn.ModuleList()

        # 生成不同的mask模式
        masks = self._create_masks(self.shape, num_layers, mask_type)

        # 创建流层
        for i in range(num_layers):
            flow = StableMaskedAffineFlow(
                shape=self.shape,
                mask=masks[i],
                hidden_dims=hidden_dims,
                init_zeros=True,
                scale_activation=scale_activation,
                scale_bound=scale_bound
            )
            self.flows.append(flow)

    def _create_masks(self, shape: Tuple, num_layers: int, mask_type: str) -> List[torch.Tensor]:
        """创建更好的mask模式"""
        total_dim = np.prod(shape)
        masks = []

        if mask_type == 'checkerboard':
            # 改进的棋盘模式
            if len(shape) >= 2:
                for i in range(num_layers):
                    mask = torch.zeros(shape)
                    for idx in range(total_dim):
                        coords = np.unravel_index(idx, shape)
                        if (sum(coords) + i) % 2 == 0:
                            flat_idx = np.ravel_multi_index(coords, shape)
                            mask.view(-1)[flat_idx] = 1
                    masks.append(mask.flatten())
            else:
                for i in range(num_layers):
                    mask = torch.zeros(total_dim)
                    mask[i % 2::2] = 1
                    masks.append(mask)

        elif mask_type == 'channel':
            # 通道分割mask
            for i in range(num_layers):
                mask = torch.zeros(total_dim)
                if i % 2 == 0:
                    mask[:total_dim // 2] = 1
                else:
                    mask[total_dim // 2:] = 1
                masks.append(mask)

        elif mask_type == 'alternating':
            # 交替mask模式
            for i in range(num_layers):
                mask = torch.zeros(total_dim)
                start_idx = i % 2
                mask[start_idx::2] = 1
                masks.append(mask)

        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

        return masks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：x -> z
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i, flow in enumerate(self.flows):
            try:
                z, log_det = flow(z)
                total_log_det += log_det

                # 检查中间结果
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print(f"NaN/Inf detected at layer {i}")
                    print(f"Z stats: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}")
                    break

            except Exception as e:
                print(f"Error at layer {i}: {e}")
                break

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播：z -> x
        """
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for i, flow in enumerate(reversed(self.flows)):
            try:
                x, log_det = flow.inverse(x)
                total_log_det += log_det

                # 检查中间结果
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"NaN/Inf detected at inverse layer {len(self.flows) - 1 - i}")
                    break

            except Exception as e:
                print(f"Error at inverse layer {len(self.flows) - 1 - i}: {e}")
                break

        return x, total_log_det

    def log_prob(self, x: torch.Tensor, base_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        计算log probability，增加数值稳定性
        """
        # 数据预处理：标准化
        x_normalized = self._normalize_input(x)

        z, log_det = self.forward(x_normalized)

        # 检查变换结果
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: NaN/Inf in transformed z")
            print(f"Input stats: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")
            print(f"Z stats: min={z[~torch.isnan(z)].min():.3f}, max={z[~torch.isinf(z)].max():.3f}")
            # 返回负无穷作为惩罚
            return torch.full((x.shape[0],), float('-inf'), device=x.device)

        if base_dist is None:
            # 使用标准正态分布
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(np.prod(self.shape), device=x.device, dtype=x.dtype),
                torch.eye(np.prod(self.shape), device=x.device, dtype=x.dtype)
            )

        # 计算基础分布的log probability
        z_flat = z.view(z.shape[0], -1)
        log_prob_z = base_dist.log_prob(z_flat)

        # 检查基础分布概率
        if torch.isnan(log_prob_z).any() or torch.isinf(log_prob_z).any():
            print("Warning: NaN/Inf in base distribution log prob")
            return torch.full((x.shape[0],), float('-inf'), device=x.device)

        return log_prob_z + log_det

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """输入标准化以提高数值稳定性"""
        # 简单的标准化
        x_mean = x.mean()
        x_std = x.std() + 1e-8
        return (x - x_mean) / x_std

    def sample(self, num_samples: int, base_dist: Optional[torch.distributions.Distribution] = None,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        从模型中采样
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(np.prod(self.shape), device=device),
                torch.eye(np.prod(self.shape), device=device)
            )

        with torch.no_grad():
            z = base_dist.sample((num_samples,)).view(num_samples, *self.shape)
            x, _ = self.inverse(z)

        return x


class MultiDimPlanarFlow(nn.Module):
    """
    支持多维tensor的Planar Flow实现

    基于论文: Variational Inference with Normalizing Flows (arXiv:1505.05770)

    变换公式: f(z) = z + u * h(w^T * z + b)
    其中:
    - z: 输入向量
    - u, w: 学习参数向量
    - b: 偏置标量
    - h: 非线性激活函数
    """

    def __init__(self,
                 shape: Union[int, Tuple],
                 activation: str = "tanh",
                 init_std: float = 0.1,
                 constraint_mode: str = "invertibility"):
        """
        初始化Planar Flow层

        Args:
            shape: 输入tensor的形状，可以是int或tuple
            activation: 激活函数类型 ("tanh", "leaky_relu", "sigmoid", "swish")
            init_std: 参数初始化的标准差
            constraint_mode: 约束模式 ("invertibility", "stability", "none")
        """
        super().__init__()

        # 处理输入形状
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

        self.total_dim = np.prod(self.shape)
        self.activation = activation
        self.constraint_mode = constraint_mode

        # 初始化参数
        self._init_parameters(init_std)

        # 设置激活函数
        self._setup_activation()

    def _init_parameters(self, init_std: float):
        """初始化参数u, w, b"""
        # w参数：权重向量
        self.w = nn.Parameter(torch.randn(self.shape) * init_std)

        # u参数：方向向量
        self.u = nn.Parameter(torch.randn(self.shape) * init_std)

        # b参数：偏置标量
        self.b = nn.Parameter(torch.zeros(1))

        # 可学习的缩放因子（用于数值稳定性）
        self.scale = nn.Parameter(torch.ones(1))

    def _setup_activation(self):
        """设置激活函数及其导数"""
        if self.activation == "tanh":
            self.h = torch.tanh
            self.h_prime = lambda x: 1 - torch.tanh(x) ** 2
        elif self.activation == "leaky_relu":
            self.negative_slope = 0.01
            self.h = nn.LeakyReLU(negative_slope=self.negative_slope)
            self.h_prime = lambda x: torch.where(x >= 0,
                                                 torch.ones_like(x),
                                                 torch.full_like(x, self.negative_slope))
        elif self.activation == "sigmoid":
            self.h = torch.sigmoid
            self.h_prime = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif self.activation == "swish":
            self.h = lambda x: x * torch.sigmoid(x)
            self.h_prime = lambda x: torch.sigmoid(x) + x * torch.sigmoid(x) * (1 - torch.sigmoid(x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _get_constrained_u(self) -> torch.Tensor:
        """
        获取受约束的u参数以确保可逆性

        约束条件: w^T * u >= -1 以保证det(∂f/∂z) > 0
        """
        w_flat = self.w.view(-1)
        u_flat = self.u.view(-1)

        # 计算w^T * u
        wu_dot = torch.sum(w_flat * u_flat)
        w_norm_sq = torch.sum(w_flat ** 2)

        if self.constraint_mode == "invertibility":
            # 严格的可逆性约束
            if w_norm_sq > 1e-8:  # 避免除零
                # 如果w^T * u < -1，则调整u
                constraint_term = (-1 - wu_dot) * w_flat / w_norm_sq
                u_constrained = u_flat + F.softplus(constraint_term.sum()) * constraint_term
            else:
                u_constrained = u_flat

        elif self.constraint_mode == "stability":
            # 更温和的稳定性约束
            if w_norm_sq > 1e-8:
                # 使用tanh来软约束
                constraint_factor = torch.tanh(wu_dot + 1)
                u_constrained = u_flat * constraint_factor
            else:
                u_constrained = u_flat

        else:  # constraint_mode == "none"
            u_constrained = u_flat

        return u_constrained.view(self.shape)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            z: 输入tensor, shape: (batch_size, *self.shape)

        Returns:
            (变换后的tensor, log determinant)
        """
        batch_size = z.shape[0]

        # 展平输入以便计算
        z_flat = z.view(batch_size, -1)
        w_flat = self.w.view(-1)

        # 获取约束后的u
        u_constrained = self._get_constrained_u()
        u_flat = u_constrained.view(-1)

        # 计算线性变换: w^T * z + b
        linear_term = torch.matmul(z_flat, w_flat) + self.b

        # 应用激活函数
        h_linear = self.h(linear_term)

        # 计算变换: z + u * h(w^T * z + b)
        # 需要广播u到batch dimension
        u_expanded = u_flat.unsqueeze(0).expand(batch_size, -1)
        h_expanded = h_linear.unsqueeze(1).expand(-1, self.total_dim)

        z_transformed_flat = z_flat + self.scale * u_expanded * h_expanded
        z_transformed = z_transformed_flat.view(batch_size, *self.shape)

        # 计算log determinant
        log_det = self._compute_log_determinant(linear_term, w_flat, u_flat)

        return z_transformed, log_det

    def _compute_log_determinant(self, linear_term: torch.Tensor,
                                 w_flat: torch.Tensor, u_flat: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobian的log determinant

        公式: log|det(∂f/∂z)| = log|1 + u^T * h'(w^T * z + b) * w|
        """
        # 计算激活函数的导数
        h_prime_val = self.h_prime(linear_term)

        # 计算u^T * w
        uw_dot = torch.sum(u_flat * w_flat)

        # 计算log determinant
        det_term = 1 + self.scale * uw_dot * h_prime_val

        # 数值稳定性：避免log(0)或log(负数)
        det_term = torch.clamp(det_term, min=1e-8)
        log_det = torch.log(torch.abs(det_term))

        return log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播（仅对LeakyReLU有解析解）

        Args:
            z: 变换后的tensor

        Returns:
            (原始tensor, log determinant)
        """
        if self.activation != "leaky_relu":
            raise NotImplementedError(
                f"Analytical inverse not available for {self.activation}. "
                "Use numerical methods or choose 'leaky_relu' activation."
            )

        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        w_flat = self.w.view(-1)

        # 获取约束后的u
        u_constrained = self._get_constrained_u()
        u_flat = u_constrained.view(-1)

        # 对于LeakyReLU，我们可以分段求解
        # 这里实现一个近似的反向传播
        # 注意：这是一个简化实现，实际应用中可能需要更精确的数值方法

        # 初始猜测
        x_flat = z_flat.clone()

        # 简单的不动点迭代（可以用更高级的方法如Newton-Raphson）
        for _ in range(10):  # 迭代次数
            linear_term = torch.matmul(x_flat, w_flat) + self.b
            h_val = self.h(linear_term)
            u_expanded = u_flat.unsqueeze(0).expand(batch_size, -1)
            h_expanded = h_val.unsqueeze(1).expand(-1, self.total_dim)

            # 更新
            x_new = z_flat - self.scale * u_expanded * h_expanded

            # 检查收敛
            if torch.max(torch.abs(x_new - x_flat)) < 1e-6:
                break
            x_flat = x_new

        x = x_flat.view(batch_size, *self.shape)

        # 计算反向log determinant
        linear_term = torch.matmul(x_flat, w_flat) + self.b
        log_det = -self._compute_log_determinant(linear_term, w_flat, u_flat)

        return x, log_det


class MultiDimPlanarFlowStack(nn.Module):
    """
    多层Planar Flow堆叠
    """

    def __init__(self,
                 shape: Union[int, Tuple],
                 num_layers: int = 8,
                 activation: str = "tanh",
                 init_std: float = 0.1,
                 constraint_mode: str = "invertibility"):
        """
        初始化多层Planar Flow

        Args:
            shape: 输入tensor形状
            num_layers: 层数
            activation: 激活函数
            init_std: 初始化标准差
            constraint_mode: 约束模式
        """
        super().__init__()

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

        self.num_layers = num_layers

        # 创建多个Planar Flow层
        self.flows = nn.ModuleList([
            MultiDimPlanarFlow(
                shape=self.shape,
                activation=activation,
                init_std=init_std,
                constraint_mode=constraint_mode
            ) for _ in range(num_layers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播通过所有层

        Args:
            z: 输入tensor

        Returns:
            (变换后的tensor, 总log determinant)
        """
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for flow in self.flows:
            z, log_det = flow(z)
            total_log_det += log_det

        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播通过所有层
        """
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        # 反向遍历所有层
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            total_log_det += log_det

        return z, total_log_det

    def log_prob(self, x: torch.Tensor,
                 base_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        计算log probability

        Args:
            x: 输入数据
            base_dist: 基础分布

        Returns:
            log probability
        """
        z, log_det = self.forward(x)

        if base_dist is None:
            # 默认使用标准正态分布
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(np.prod(self.shape), device=x.device, dtype=x.dtype),
                torch.eye(np.prod(self.shape), device=x.device, dtype=x.dtype)
            )

        z_flat = z.view(z.shape[0], -1)
        log_prob_z = base_dist.log_prob(z_flat)

        return log_prob_z + log_det

    def sample(self, num_samples: int,
               base_dist: Optional[torch.distributions.Distribution] = None,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        从模型中采样

        Args:
            num_samples: 采样数量
            base_dist: 基础分布
            device: 设备

        Returns:
            采样的数据
        """
        if base_dist is None:
            base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(np.prod(self.shape), device=device),
                torch.eye(np.prod(self.shape), device=device)
            )

        with torch.no_grad():
            z = base_dist.sample((num_samples,)).view(num_samples, *self.shape)

            if self.flows[0].activation == "leaky_relu":
                x, _ = self.inverse(z)
                return x
            else:
                # 对于没有解析逆的激活函数，我们只能从forward方向采样
                print("Warning: Analytical inverse not available. Returning samples from base distribution.")
                return z


def create_target_distribution(n_samples=1000):
    """创建目标复杂分布数据"""
    # 创建一个双峰分布
    # 第一个峰
    mean1 = torch.tensor([-1.5, -1.5])
    cov1 = torch.tensor([[0.3, 0.1], [0.1, 0.3]])
    dist1 = MultivariateNormal(mean1, cov1)

    # 第二个峰
    mean2 = torch.tensor([1.5, 1.5])
    cov2 = torch.tensor([[0.3, -0.1], [-0.1, 0.5]])
    dist2 = MultivariateNormal(mean2, cov2)

    # 混合采样
    samples1 = dist1.sample((n_samples // 2,))
    samples2 = dist2.sample((n_samples // 2,))

    samples = torch.cat([samples1, samples2], dim=0)

    # 数据标准化
    samples = (samples - samples.mean(dim=0)) / (samples.std(dim=0) + 1e-8)

    return samples


def train_normalizing_flow():
    """训练标准化流模型"""
    # 超参数
    dim = 2
    n_flows = 4 # 减少流层数
    n_epochs = 2600
    batch_size = 128  # 减少批次大小
    learning_rate = 5e-4  # 降低学习率

    # 创建目标数据
    target_data = create_target_distribution(3000)
    print(f"目标数据形状: {target_data.shape}")
    print(f"目标数据范围: [{target_data.min():.3f}, {target_data.max():.3f}]")

    # 创建模型
    # model = MultiDimRealNVP(dim, num_layers=n_flows, mask_type='channel')
    model = MultiDimPlanarFlowStack(
        shape=dim,
        num_layers=n_flows,
        activation="tanh",
        constraint_mode="invertibility"
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8)

    # 训练循环
    losses = []

    for epoch in range(n_epochs):
        # 随机批次
        idx = torch.randperm(len(target_data))[:batch_size]
        batch_data = target_data[idx]

        try:
            # 计算负对数似然损失
            log_prob = model.log_prob(batch_data)
            loss = -log_prob.mean()

            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"第{epoch}轮损失无效，跳过更新")
                continue

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(loss)

            losses.append(loss.item())

            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        except Exception as e:
            print(f"第{epoch}轮训练出错: {e}")
            continue

    return model, losses, target_data


def visualize_results(model, target_data, losses):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 训练损失
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Negative Log Likelihood')
    axes[0, 0].grid(True)

    # 2. 原始复杂分布
    axes[0, 1].scatter(target_data[:, 0].detach().numpy(), target_data[:, 1].detach().numpy(), alpha=0.6, s=1)
    axes[0, 1].set_title('Target Complex Distribution')
    axes[0, 1].set_xlabel('X1')
    axes[0, 1].set_ylabel('X2')
    axes[0, 1].grid(True)

    # 3. 变换后的分布（应该接近标准高斯）
    try:
        with torch.no_grad():
            z_transformed, _ = model.forward(target_data)
        axes[0, 2].scatter(z_transformed[:, 0].detach().numpy(), z_transformed[:, 1].detach().numpy(), alpha=0.6, s=1)
        axes[0, 2].set_title('Transformed to Standard Gaussian')
        axes[0, 2].set_xlabel('Z1')
        axes[0, 2].set_ylabel('Z2')
        axes[0, 2].grid(True)
        mardia_test(z_transformed)
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'变换出错: {str(e)}', transform=axes[0, 2].transAxes)

    # 4. 从模型采样的分布
    try:
        with torch.no_grad():
            generated_samples = model.sample(1000)
        axes[1, 0].scatter(generated_samples[:, 0].detach().numpy(), generated_samples[:, 1].detach().numpy(),
                           alpha=0.6, s=1)
        axes[1, 0].set_title('Generated Samples')
        axes[1, 0].set_xlabel('X1')
        axes[1, 0].set_ylabel('X2')
        axes[1, 0].grid(True)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'采样出错: {str(e)}', transform=axes[1, 0].transAxes)

    # 5. 标准高斯分布对比
    standard_gaussian = torch.randn(1000, 2)
    axes[1, 1].scatter(standard_gaussian[:, 0].detach().numpy(), standard_gaussian[:, 1].detach().numpy(), alpha=0.6,
                       s=1)
    axes[1, 1].set_title('Standard Gaussian')
    axes[1, 1].set_xlabel('Z1')
    axes[1, 1].set_ylabel('Z2')
    axes[1, 1].grid(True)

    # 6. 概率密度等高线图
    try:
        x_range = torch.linspace(-3, 3, 30)
        y_range = torch.linspace(-3, 3, 30)
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)

        with torch.no_grad():
            log_probs = model.log_prob(points)
            probs = torch.exp(log_probs).reshape(30, 30)

        axes[1, 2].contour(X.detach().numpy(), Y.detach().numpy(), probs.detach().numpy())
        axes[1, 2].set_title('Learned Probability Density')
        axes[1, 2].set_xlabel('X1')
        axes[1, 2].set_ylabel('X2')
        axes[1, 2].grid(True)
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'密度图出错: {str(e)}', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.show()


def test_flow_transformations(model, n_test_points=5):
    """测试流变换的双向性"""
    print("\n=== 测试流变换 ===")

    # 生成测试点
    test_points = torch.randn(n_test_points, 2) * 2
    model.eval()

    with torch.no_grad():
        for i, point in enumerate(test_points):
            point = point.unsqueeze(0)  # 添加批次维度

            # 前向变换
            z, log_det_forward = model.forward(point)

            # 逆变换
            x_reconstructed, log_det_inverse = model.inverse(z)

            # 计算重构误差
            reconstruction_error = torch.norm(point - x_reconstructed, p=2).item()

            # 计算概率密度
            log_prob = model.log_prob(point)

            print(f"\n测试点 {i + 1}:")
            print(f"  原始点: [{point[0, 0].item():.4f}, {point[0, 1].item():.4f}]")
            print(f"  变换后: [{z[0, 0].item():.4f}, {z[0, 1].item():.4f}]")
            print(f"  重构点: [{x_reconstructed[0, 0].item():.4f}, {x_reconstructed[0, 1].item():.4f}]")
            print(f"  重构误差: {reconstruction_error:.6f}")
            print(f"  前向log_det: {log_det_forward.item():.4f}")
            print(f"  逆向log_det: {log_det_inverse.item():.4f}")
            print(f"  log_det总和: {(log_det_forward + log_det_inverse).item():.6f}")
            print(f"  对数概率密度: {log_prob.item():.4f}")
            print(f"  概率密度: {torch.exp(log_prob).item():.6f}")


def demonstrate_distribution_mapping(model, target_data):
    """演示分布映射过程"""
    print("\n=== 分布映射演示 ===")

    # 选择一些代表性点
    representative_points = target_data[:10]

    with torch.no_grad():
        print("原始复杂分布 -> 标准高斯分布映射:")
        print("原始点\t\t\t变换后点\t\t对数概率密度")
        print("-" * 70)

        for i, point in enumerate(representative_points):
            point = point.unsqueeze(0)

            try:
                z, _ = model.forward(point)
                log_prob = model.log_prob(point)

                print(f"[{point[0, 0].item():6.3f}, {point[0, 1].item():6.3f}] -> "
                      f"[{z[0, 0].item():6.3f}, {z[0, 1].item():6.3f}] "
                      f"log_p = {log_prob.item():8.4f}")

            except Exception as e:
                print(f"点 {i} 处理出错: {e}")


import scipy.stats as stats
import math

def mardia_test(tensor, alpha=0.05):
    """Mardia多元正态性检验"""
    n, p = tensor.shape

    # 中心化数据
    mean = torch.mean(tensor, dim=0)
    centered = tensor - mean

    # 计算协方差矩阵
    cov = torch.cov(centered.T)
    cov_inv = torch.linalg.inv(cov)

    # 计算马氏距离
    mahal_dist = torch.sum(centered @ cov_inv * centered, dim=1)

    # Mardia偏度检验
    skewness = torch.mean(mahal_dist ** 3)
    skew_stat = n * skewness / 6
    skew_p_value = 1 - stats.chi2.cdf(skew_stat.item(), df=p * (p + 1) * (p + 2) / 6)

    # Mardia峰度检验
    kurtosis = torch.mean(mahal_dist ** 2)
    kurt_stat = (kurtosis - p * (p + 2)) / math.sqrt(8 * p * (p + 2) / n)
    kurt_p_value = 2 * (1 - stats.norm.cdf(abs(kurt_stat.item())))

    print(
        f"skewness_p_value: {skew_p_value:6.3f}"
        f"kurtosis_p_value: {kurt_p_value:6.3f}"
        f"is_normal: {skew_p_value > alpha and kurt_p_value > alpha}"
    )

def pca_projection(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    PCA降维

    Args:
        data: 输入数据
        n_components: 目标维度

    Returns:
        降维后的数据
    """
    pca = PCA(n_components=n_components, random_state=42)
    data_2d = pca.fit_transform(data)

    # 保存PCA模型和统计信息
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA解释方差比: {explained_variance}")
    print(f"累计解释方差: {np.sum(explained_variance):.3f}")

    return data_2d


# # 主函数
# if __name__ == "__main__":
#     print("开始训练标准化流模型...")
#
#     # 设置随机种子
#     torch.manual_seed(42)
#     np.random.seed(42)
#
#     # 训练模型
#     model, losses, target_data = train_normalizing_flow()
#
#     print("训练完成，开始可视化结果...")
#
#     # 可视化结果
#     visualize_results(model, target_data, losses)
#
#     # 测试流变换
#     # test_flow_transformations(model)
#
#     # 演示分布映射
#     demonstrate_distribution_mapping(model, target_data)
#
#     # 保存模型
#     try:
#         torch.save(model.state_dict(), 'normalizing_flow_model.pth')
#         print("\n模型已保存为 'normalizing_flow_model.pth'")
#     except Exception as e:
#         print(f"保存模型时出错: {e}")
#
#     print("\n=== 总结 ===")
#     print(f"训练轮数: {len(losses)}")
#     print(f"最终损失: {losses[-1]:.4f}")
#     print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")



# 主函数
if __name__ == "__main__":

    h_list = torch.load("h_list.pt")
    label_list = torch.load("label_list.pt")
    z_list = torch.load("z_list.pt")
    h_list = torch.cat(h_list)
    z_list = torch.cat(z_list)
    h_list=pca_projection(h_list.detach().numpy(), n_components=2)
    z_list= pca_projection(z_list.detach().numpy(), n_components=2)
    label_list = torch.cat(label_list)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # x = np.linspace(h_list.min(), h_list.max(), 100)
    # y=x

    # 2. 原始复杂分布
    axes[0].scatter(z_list[:, 0], z_list[:, 1], alpha=0.6, s=1,c=label_list,cmap='coolwarm')
    axes[0].set_title('Target Complex Distribution')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True)

    axes[1].scatter(z_list[:, 0], z_list[:, 1], alpha=0.6, s=1,c=label_list,cmap='coolwarm')
    axes[1].set_title('Transformed to Standard Gaussian')
    axes[1].set_xlabel('Z1')
    axes[1].set_ylabel('Z2')
    axes[1].grid(True)
    #
    # # 5. 标准高斯分布对比
    # standard_gaussian = torch.randn(1000, 2)
    # axes[2].scatter(standard_gaussian[:, 0].detach().numpy(), standard_gaussian[:, 1].detach().numpy(), alpha=0.6,
    #                    s=1)
    # axes[2].set_title('Standard Gaussian')
    # axes[2].set_xlabel('Z1')
    # axes[2].set_ylabel('Z2')
    # axes[2].grid(True)

    plt.tight_layout()
    plt.show()
