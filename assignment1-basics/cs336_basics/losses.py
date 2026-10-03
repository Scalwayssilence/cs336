import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    
    参数:
        logits: 预测的分数，形状为 (..., vocab_size)
        targets: 目标 ID，形状为 (...)
        
    返回:
        平均损失标量。
    """
    # 1. 提取维度信息
    # 假设最后一维是 vocab_size，前面所有的维度都是 batch-like 维度
    # 在 Transformer 训练中，输入通常是 (Batch, Seq_Len, Vocab)
    if logits.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("targets must match logits.shape[:-1] and be non-empty")
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    
    # 2. 数值稳定性：计算每组 Logits 的最大值 M
    # dim=-1 表示在词表维度找最大，keepdim=True 方便后续广播减法 [1,2,3] -> 3 -> [3]
    # m: (Batch, Seq, 1) 
    m = torch.max(logits, dim=-1, keepdim=True).values
    
    # 3. 提取减去最大值后的目标分数，避免大数相减丢失精度。
    # 使用 gather 函数从 logits 中根据 targets 提取对应的分值
    # logits: (Batch, Seq, Vocab) -> targets: (Batch, Seq)
    # 我们需要将 targets 升维成 (Batch, Seq, 1) 才能用 gather
    shifted_logits = logits - m
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # 4. 计算 Log-Sum-Exp 项
    # log(sum(exp(o - M))) - (o_y - M)，公共偏移 M 已消去。
    # 注意：为了防止 sum 结果为 0 导致 log(-inf)，这里的减法已经保证了 exp 的最大值是 e^0 = 1
    # shifted_logits: (Batch, Seq, Vocab) 
    # log_sum_exp:(Batch, Seq) 
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    
    # 5. 计算单个 Token 的损失: log_sum_exp - o_y
    loss = log_sum_exp - target_logits
    
    # 6. PDF 要求返回整个 Batch 的平均值，标量
    return torch.mean(loss)
