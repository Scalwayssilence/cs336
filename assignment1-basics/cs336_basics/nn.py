import torch
import torch.nn as nn
import math
from einops import rearrange

# =============================================================================
# Transformer 学习笔记：张量维度与阅读顺序
# =============================================================================
# B = batch size：一批序列的数量；S = sequence length：每条序列的 token 数。
# H = num_heads：注意力头数；D = d_model：模型维度；d_k = D / H：每头 Q/K 维度。
# d_ff：FFN 中间维度；vocab_size：词表大小；token ID：词表中的整数索引。
# 例：D=512、H=8，则 Q/K/V 从 [B,S,512] 拆成 [B,8,S,64]。
# Attention 汇总不同 token 的信息；FFN 加工每个 token 自己的特征。
# 入口：TransformerLM.forward；生成入口：TransformerLM.generate。
# __init__ 创建参数和子模块；调用 module(x) 时执行该模块的 forward。
# 文件末尾有完整模型、Pre-Norm Block、注意力和 SwiGLU 的架构图。


# --- Linear：线性投影 ---------------------------------------------------------
# y = x @ W.T；W 的形状为 [out_features,in_features]，本实现不含偏置。
# 只改变最后一维：[...,in_features] -> [...,out_features]。
# nn.Parameter 注册可训练权重；device 指 CPU/GPU，dtype 指数值精度。


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 1. 定义权重 W (形状: out x in)
        # 注意要把 device 和 dtype 传进去，确保张量创建在正确位置
        factory_kwargs = {'device': device, 'dtype': dtype}

        # nn.Parameter 的作用是告诉 PyTorch：“这个张量是模型的一部分，它是需要通过训练来学习的权重（Weights）。”
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 2. 初始化权重 (截断正态分布)  Xavier 初始化
        # 根据 3.4.1: sigma^2 = 2 / (din + dout)
        std = (2.0 / (in_features + out_features)) ** 0.5
        # PDF 要求截断在 [-3sigma, 3sigma]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 einsum 处理，适应各种 Batch 维度情况
        # '...i' 表示输入 x 的最后一个维度 (in_features)
        # 'oi' 表示权重 W (out_features, in_features)
        # '-> ...o' 表示输出保留前面的维度，最后一个维度变成 out_features
        return torch.einsum('...i, oi -> ...o', x, self.weight)


# --- Embedding：token ID 查表 -------------------------------------------------
# 权重表 [vocab_size,D] 的每一行是一个 token 的可学习向量。
# 输入整数 ID [B,S]，输出向量 [B,S,D]；本层不加入位置信息。
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# --- RMSNorm：归一化 ----------------------------------------------------------
# 作用：控制特征尺度，帮助训练稳定；对每个 token 的最后一维独立计算。
# y = x / sqrt(mean(x^2) + eps) * g；g 是可学习缩放，初始为 1。
# LayerNorm：先减均值、再除标准差，通常还有缩放和偏移。
# RMSNorm：不减均值，只按均方根缩放；输出形状不变。
# eps 防止除零；这里先用 float32 计算，再恢复输入类型。
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 1. 必须初始化为全 1 (ones)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)

        in_dtype = x.dtype
        # 2. 转换为 float32 以防平方计算时溢出
        x_float = x.to(torch.float32)

        # 3. 计算均方根 (Root Mean Square)
        # 公式: rms = sqrt( mean(x^2) + eps )
        # dim=-1 表示在隐藏层维度计算，keepdim=True 方便后续除法自动广播
        # 广播按形状右对齐：对应维度相等或一方为 1；缺失的左侧维度视为 1。
        # keepdim=True 保留 [B,S,1]，可与输入 [B,S,D] 逐元素相除。
        # 广播是逻辑扩展，不代表必须实际复制整个张量。
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 归一化并乘以可学习的增益参数 g
        result = (x_float / rms) * self.weight

        # 5. 转回原始类型
        return result.to(in_dtype)


# --- SiLU / Swish(beta=1)：平滑激活 -------------------------------------------
# SiLU(z)=z*sigmoid(z)；输出不局限于 [0,1]，可为负，也可大于 1。
def silu_fn(in_features):
    return in_features * torch.sigmoid(in_features)


# --- SwiGLU：带门控的前馈网络 -------------------------------------------------
# 两路输入投影：gate = SiLU(w1(x))，signal = w3(x)。
# 输出：w2(gate * signal)；* 是逐元素乘法，门控可缩小、放大或改变符号。
# 形状：[B,S,D] -> 两路 [B,S,d_ff] -> 相乘 -> [B,S,D]。
# 各 token 使用同一套参数，但 FFN 不直接混合不同位置的信息。
# 8/3 准则（忽略偏置）：普通 FFN 的 D->4D->D 有 8D^2 个参数；
# SwiGLU 有三个矩阵，共 3D*d_ff 个参数，匹配预算得 d_ff=(8/3)D。
# 这也等于普通 FFN 中间维度 4D 的 2/3；实际可取整到硬件友好的倍数。
# 注意：本类不自动计算 8/3，d_ff 由调用方传入。
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # W1 和 W3 是并行升维层: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        # 形状: [..., d_ff]
        return self.w2(gate * signal)


# --- RoPE：旋转位置编码 -------------------------------------------------------
# 内容匹配本身不直接提供位置；RoPE 在 Q/K 生成后，按位置旋转相邻维度对。
# 第 r 对频率 omega_r = theta^(-2r/d_k)，角度 angle = position * omega_r。
# r 从 0 开始；前面的维度对转得快，后面的转得慢，提供多尺度位置模式。
# 对每一对 (a,b)：a' = a*cos(angle)-b*sin(angle)，b' = a*sin(angle)+b*cos(angle)。
# 相对位置性质：(R_m q).T @ (R_n k) = q.T @ R_(n-m) @ k。
# 因此位置因素依赖距离 n-m，但实际注意力分数仍取决于 q/k 的内容。
# 标准用法只旋转 Q/K，不旋转 V；d_k 必须为偶数，输出形状不变。
# 本实现用切片和逐元素运算代替大旋转矩阵，缓存不是可训练参数。
# 距离增加不保证每个分数单调下降；能计算长位置也不保证可靠外推。
# 缓存只覆盖 [0,context_length)，传入位置须在范围内。
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, context_length: int, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        context_length: 最大序列长度
        """
        super().__init__()
        self.d_k = d_k

        # 1. 计算频率频率 omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是成对进行的
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)

        # 2. 创建位置序列 [0, 1, ..., context_length - 1]
        t = torch.arange(context_length, device=device).float()  # 形状: (context_length,)

        # 3. 计算所有位置的所有角度 (外积)
        # freqs_matrix 形状: (context_length, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    # 位置查表 -> 对齐 batch/head 维 -> 成对旋转，保持输入形状和 dtype。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions=[B,S] 时，查表结果为 [B,S,d_k/2]。
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 常见输入 [B,H,S,d_k]：将缓存 [B,S,d_k/2] 变成 [B,1,S,d_k/2]。
        # 若位置是 [S]，缓存 [S,d_k/2] 可以自动广播，不需插入维度。
        # 插入位置取决于布局；若输入为 [B,S,H,d_k]，需在 S 后插入头维。
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output


# --- Softmax：分数转权重 ------------------------------------------------------
# p_i=exp(x_i)/sum(exp(x_j))；沿 dim 归一化，权重非负且和为 1。
# 先减最大值不改变结果，但可避免 exp 溢出；dim=-1 表示最后一维。
# 注意力中最后一维是 Key 的位置，LM 输出中则是词表。
# 注意：若一整行都是 -inf，本实现会产生 NaN，因此不可屏蔽一整行。
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 1. 为了数值稳定性，减去指定维度上的最大值
    # dim 决定哪一组分数被归一化，具体含义由调用方指定。
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    # 2. 计算指数
    exp_x = torch.exp(x_stable)

    # 3. 计算分母的各指数之和
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # 4. 计算最终结果
    return exp_x / sum_exp


# --- 缩放点积注意力：Q/K 匹配，V 提供内容 -------------------------------------
# Q（Query）：要找什么；K（Key）：用什么特征匹配；V（Value）：实际汇总的信息。
# Attention(Q,K,V)=softmax(Q@K.T/sqrt(d_k)+mask)@V。
# 除 sqrt(d_k) 控制点积分数尺度；softmax 在 Key 位置上计算。
# mask=True 表示允许关注，False 对应分数 -inf，权重变为 0。
# 权重例 [0.1,0.7,0.2]：输出 0.1*v1+0.7*v2+0.2*v3，并非只挑一个 token。
def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    d_k = Q.size(-1)

    # 1. 计算分数: Q @ K^T / sqrt(d_k)
    # 交换最后两个维度进行矩阵乘法
    # 形状变化: (..., n, d_k) @ (..., d_k, m) -> (..., n, m)
    scores = torch.einsum('...nk, ...mk-> ...nm', Q, K) / math.sqrt(d_k)

    # 2. 应用掩码
    if mask is not None:
        # PDF 要求: 把 mask 为 False 的地方填入 -inf
        # -inf 对应 exp(-inf)=0；每一行须至少保留一个可关注位置。
        scores = scores.masked_fill(mask == False, float('-inf'))

    # 3. Softmax 归一化 (在最后一个维度 m 上)
    # 注意: 这里的 dim=-1 指向的是 Key 序列的长度维度
    probs = softmax(scores, dim=-1)

    # 4. 对 Value 加权求和
    # 形状变化: (..., n, m) @ (..., m, d_v) -> (..., n, d_v)
    output = torch.einsum('...nm, ...mk-> ...nk', probs, V)

    return output


# --- 因果多头自注意力 ---------------------------------------------------------
# 自注意力：Q/K/V 都来自同一个输入序列；每个 token 按权重汇总上下文。
# 多头：多组可学习投影并行计算，能学习不同关系，分工不由人为固定。
# 本实现先投影到 D，再拆成 H 个头；要求 D 能被 H 整除，d_k=D/H。
# 多头结果拼接后做输出投影，恢复 [B,S,D]，方便残差相加。
# 因果遮罩：第 i 个位置只能看 j<=i，防止预测时读取未来 token。
# RoPE 可选；theta/context_length 都提供时启用。
# 输入 [B,S,D] -> Q/K/V [B,H,S,d_k] -> 分数 [B,H,S,S] -> 输出 [B,S,D]。
# bias 参数当前未参与计算，投影均使用上面的无偏置 Linear。
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = False,
                 context_length=None, theta=None,
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 1. 定义 Q, K, V 的投影层（PDF 要求 3 次矩阵乘法）
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 2. 定义输出投影
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 3. 实例化 RoPE
        if theta is not None and context_length is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, context_length, device=device)
        else:
            self.rope = None

    # 投影拆头 -> Q/K 的 RoPE -> 因果注意力 -> 合头和输出投影。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.shape

        # (h d) 将模型维度拆成头数和每头维度，再把头维移到序列维前。
        # [B,S,D] -> [B,H,S,d_k]；不是给每个头复制完整的 D 维向量。
        q = rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)

        # 步骤 3: 应用 RoPE
        # 只有当模块存在时才应用
        if self.rope is not None:
            # 如果没传位置，且 RoPE 需要位置，则生成默认位置
            if token_positions is None:
                # 本入口 x 为 [B,S,D]；expand 为批次共享默认位置，避免显式复制。
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 生成下三角矩阵
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        # 步骤 5: SDPA (SDPA 内部应能处理 mask 为 None 的情况)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 步骤 6 & 7: 合并与输出投影
        attn_out = rearrange(attn_out, '... h s d -> ... s (h d)')
        return self.output_proj(attn_out)


import torch
import torch.nn as nn
from .nn import Embedding, RMSNorm, Linear, CausalSelfAttention, SwiGLU


# --- Transformer Block：归一化、注意力、FFN 与残差 ----------------------------
# 残差连接 y=x+F(x)：在已有表示上学习调整，要求两路形状相同，不是拼接。
# 若 F(x)≈0，则 y≈x；梯度含直接路径 I，有助于深层训练，但不是稳定性保证。
# Pre-Norm：u=x+Attention(Norm1(x))；y=u+FFN(Norm2(u))。
# Post-Norm：u=Norm1(x+Attention(x))；y=Norm2(u+FFN(u))。
# 一个 Block 有两次残差相加；第二次的输入是第一次更新后的 u。
# ln1/ln2 实际为两个独立 RMSNorm；禁用时 Identity(x)=x。
# 注意：norm_mode 只支持 pre/post，其他值目前会直接返回原输入。
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int,
                 theta: float, device=None, dtype=None,
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",  # 选项: "pre", "post"
                 ffn_type: str = "swiglu"  # 选项: "swiglu", "silu"
                 ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        # 1. 初始化 Attention
        # RoPE 的开关由外部传入的 theta 是否为 None 控制
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            theta=theta,
            device=device,
            dtype=dtype
        )

        # 2. 初始化 Norm 层 (Ablation 1)
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            # 如果禁用 Norm，使用 Identity 占位，它直接返回输入，不改变任何东西
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # 3. 初始化 FFN (Ablation 4)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # 标准 FFN: x -> Linear -> SiLU -> Linear -> out
            # 注意: 为了公平对比，通常 SiLU FFN 的 d_ff 应该是 4 * d_model
            d_ff = 4 * d_model
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    # 两个子层分别做一次残差更新；x=x+... 是重新赋值，不是原地 x+=...。
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm (Llama 默认, 也是作业基准)
        # 公式: x = x + Sublayer(Norm(x))
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))

        # Post-norm (原始 Transformer, Ablation 2)
        # 公式: x = Norm(x + Sublayer(x))
        elif self.norm_mode == "post":
            # 先残差相加，再归一化；训练稳定性也取决于初始化和学习率等设置。
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))

        return x


# --- Transformer 语言模型 ----------------------------------------------------
# token ID -> Embedding -> 多个 Block -> Final Norm -> LM Head -> logits。
# logits 是每个词表 token 的原始分数，不是概率；softmax 后才得到概率。
# 输入 [B,S]，输出 [B,S,vocab_size]；位置 s 的输出用于预测其后一个 token。
# 各 Block 有独立参数；默认使用 Pre-Norm、RMSNorm 和 SwiGLU。
# 本文件定义模型与生成过程，训练时的损失和优化步骤由外部代码负责。
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None, dtype=None,
                 # 新增实验参数
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.context_length = context_length

        # 1. Token Embedding 层
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2. 堆叠 Transformer Blocks
        # 将实验参数透传给每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta,
                device=device, dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])

        # 3. 最终的输出层
        # 如果全局禁用了 Norm，这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            """
            forward(input):
                return input
            """
            self.ln_final = nn.Identity()

        # 最后是一个 Linear 层映射回词表大小 (LM Head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    # 为每条序列生成位置，逐层更新表示，最后返回所有位置的词表 logits。
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        b, s = token_ids.shape

        # 准备位置信息用于 RoPE, shape: [S] -> [1, S] -> [B,S]
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        # 1. Embedding
        x = self.token_embeddings(token_ids)

        # 2. 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 3. 最终归一化 (如果 use_rms_norm=False，这里就是直通)
        x = self.ln_final(x)

        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)

    # 自回归生成：取最后位置的 logits -> 温度/Top-P -> softmax -> 采样 -> 追加。
    # 每次重算最近 context_length 个 token；这里没有 KV 缓存。
    # no_grad 禁用梯度记录；eval 切到评估模式，但不会自动恢复先前模式。
    # 当前 EOS 条件是同一步所有批次都生成 EOS，不是逐条记录完成状态。
    @torch.no_grad()
    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            eos_token_id: int = None,
            temperature: float = 1.0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """
        从模型生成文本 ID 序列。

        参数:
            prompt_ids: 提示词 ID (Batch, Seq_len)
            max_new_tokens: 最多生成的词数
            eos_token_id: 停止生成的 Token ID (如 <|endoftext|>)
            temperature: 温度系数 (越高越随机，越低越确定)
            top_p: 核采样阈值
        """
        # 设置为评估模式
        self.eval()

        # 将输入拷贝一份，避免修改原始数据
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # 1. 裁剪输入：模型只能处理 context_length 长度的内容
            # 如果生成的序列过长，只取最后的 context_length 个词
            idx_cond = generated[:, -self.context_length:]

            # 2. 前向传播得到 Logits
            # 我们只关心最后一个时间步的预测
            logits = self.forward(idx_cond)  # (Batch, T, Vocab)
            logits = logits[:, -1, :]  # (Batch, Vocab)

            # 3. 应用温度 (Temperature)
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8)  # 加个 epsilon 防止除以 0

            # 4. 应用 Top-P (Nucleus Sampling) 过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)

            # 5. 归一化并采样
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (Batch, 1)

            # 6. 拼接新词
            generated = torch.cat((generated, next_token), dim=1)

            # 7. 如果遇到了 EOS，提前结束生成
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # Top-P：按概率降序保留累计概率达到 p 的最小前缀，其余设为 -inf。
    # 掩码右移一格，保留首次超过阈值的 token，并保证至少留下一个候选。
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 计算累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # 创建掩码：我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True（需要移除）
        sorted_indices_to_remove = cumulative_probs > p

        # 关键修正：确保至少保留第一个词（最高概率词），
        # 并且我们要保留第一个“使概率超过 p”的那个词。
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits


# =============================================================================
# 架构图：默认 Pre-Norm + RMSNorm + SwiGLU（RoPE 启用时）
# =============================================================================
# 1. TransformerLM：完整前向传播
#
#    token_ids [B,S]       positions [B,S]
#           |                    |
#       Embedding                |  每层传给 Q/K 的 RoPE
#           |                    |
#       x [B,S,D]                |
#           |                    |
#    TransformerBlock <----------+
#           |  重复 num_layers 次，各层参数独立
#       x [B,S,D]
#           |
#       Final RMSNorm
#           |
#       LM Head: Linear(D,vocab_size)
#           |
#    logits [B,S,vocab_size]
#           |
#    生成时：取最后位置 -> 温度/Top-P -> softmax -> 采样 -> 追加 token
#
# 2. TransformerBlock：两条残差旁路，形状始终为 [B,S,D]
#
#    x --------+----------------------------------+
#              |                                  |
#           RMSNorm1                              |
#              |                                  |
#       CausalSelfAttention                       |
#              |                                  |
#              +---------------- (+) <------------+
#                                 |
#    u ---------------------------+---------------+
#                                 |               |
#                              RMSNorm2           |
#                                 |               |
#                               SwiGLU            |
#                                 |               |
#                                 +-- (+) <-------+
#                                      |
#                                      y
#
#    Pre-Norm : u=x+Attn(Norm1(x)); y=u+FFN(Norm2(u))
#    Post-Norm: u=Norm1(x+Attn(x)); y=Norm2(u+FFN(u))
#    禁用归一化时，上述 Norm 都变成 Identity。
#
# 3. CausalSelfAttention：Q/K 匹配，V 提供内容
#
#                       x [B,S,D]
#                            |
#               +------------+------------+
#               |            |            |
#            q_proj       k_proj       v_proj
#               |            |            |
#            拆成 H 头，每路 [B,H,S,d_k]
#               |            |            |
#             RoPE         RoPE           V
#               |            |            |
#               Q            K            |
#               +------ Q @ K.T           |
#                          |              |
#                    / sqrt(d_k)          |
#                          |              |
#                因果 mask: j<=i          |
#                          |              |
#               softmax [B,H,S,S]         |
#                          |              |
#                          +----- @ V <---+
#                                   |
#                         output [B,H,S,d_k]
#                                   |
#                           合头 [B,S,D]
#                                   |
#                             output_proj
#                                   |
#                             输出 [B,S,D]
#
# 4. SwiGLU：FFN 内部是乘法，Block 外部的残差是加法
#
#                      x [B,S,D]
#                           |
#                 +---------+---------+
#                 |                   |
#                w1                  w3
#                 |                   |
#               SiLU               signal
#                 |                   |
#                gate ------ (*) <----+
#                             |
#                       h [B,S,d_ff]
#                             |
#                            w2
#                             |
#                       输出 [B,S,D]
#
#    图中 (+) = 对应元素相加，(*) = 对应元素相乘，@ = 矩阵乘法。
