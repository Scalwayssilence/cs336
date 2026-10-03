import argparse
import os
import json
import torch
import numpy as np
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW, clip_gradient_norm
from cs336_basics.scheduler import get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.losses import cross_entropy


def main(argv=None):
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    
    # --- 实验/消融 (Ablation) 开关 ---
    # Ablation 1: 移除 RMSNorm
    parser.add_argument("--no_rms_norm", action="store_true", help="Disable RMSNorm completely")
    # Ablation 2: Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type=str, default="pre", choices=["pre", "post"], help="Normalization placement")
    # Ablation 3: 移除 RoPE (NoPE)
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings")
    # Ablation 4: SwiGLU vs SiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Type of Feed-Forward Network")

    # --- 优化器超参数 ---
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, help="显式指定要恢复的 checkpoint")
    parser.add_argument("--wandb", action="store_true", help="启用可选的 WandB 日志上传")
    
    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")
    
    args = parser.parse_args(argv)
    for name in ('batch_size', 'context_length', 'd_model', 'num_layers', 'num_heads',
                 'd_ff', 'vocab_size', 'max_iters', 'eval_interval', 'eval_batches', 'save_interval'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.d_model % args.num_heads or (not args.no_rope and (args.d_model // args.num_heads) % 2):
        parser.error('d_model must be divisible by num_heads; RoPE requires an even head dimension')
    get_lr_cosine_schedule(0, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
    if args.max_norm < 0:
        parser.error('max_norm must be non-negative')
    torch.manual_seed(args.seed)
    # 默认仅在本地输出日志；指定 --wandb 才加载并上传。
    wandb_run = None

    if not args.resume and os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):
        raise FileExistsError('out_dir already contains ckpt.pt; use --resume or a new out_dir')

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据 (使用 memmap)
    # 假设数据是以 uint16 存储的二进制文件
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    # np.memmap 延迟加载数据到内存，非常适合大数据集，并且将二进制文件转为 dtype （uint16） 数组
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')
    for name, data in [('train', train_data), ('valid', val_data)]:
        if len(data) <= args.context_length:
            raise ValueError(f'{name}: at least context_length + 1 tokens required')
        # 分块检查，避免为大型语料创建完整副本。
        for start in range(0, len(data), 1_000_000):
            if int(data[start:start + 1_000_000].max()) >= args.vocab_size:
                raise ValueError(f'{name}: token ID exceeds vocab_size')

    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    # 2. 处理消融实验逻辑
    # 如果 no_rope 为 True，则 theta 设为 None，TransformerBlock 内部就不会初始化 RoPE
    actual_rope_theta = None if args.no_rope else 10000.0
    # use_rms_norm 逻辑取反
    use_rms_norm = not args.no_rms_norm

    # 3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        d_model=args.d_model, 
        num_layers=args.num_layers,
        num_heads=args.num_heads, 
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        device=args.device,
        # 传入实验参数
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)

    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}, RoPE={not args.no_rope}")

    # 4. 初始化优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from iteration {start_iter}")
    if start_iter >= args.max_iters:
        raise ValueError('max_iters must exceed the completed checkpoint iteration')

    # 6. 保存配置；只有显式启用时才上传 WandB。
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    with open(os.path.join(args.out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 7. 主训练循环
    for it in range(start_iter, args.max_iters):
        # A. 更新学习率
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        logits = model(x)
        loss = cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite training loss at iteration {it}')
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)
        
        optimizer.step()

        # C. 验证与日志记录
        if (it + 1) % args.eval_interval == 0 or it == args.max_iters - 1:
            model.eval()
            # 验证采样不改变后续训练采样的随机状态。
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                v_loss = torch.zeros((), device=args.device)
                for _ in range(args.eval_batches):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    v_loss += cross_entropy(model(vx), vy) / args.eval_batches
                print(f"Iter {it + 1}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                metrics = {
                    "train/loss": loss.item(), 
                    "val/loss": v_loss.item(), 
                    "lr": lr, 
                    "iter": it + 1
                }
                with open(os.path.join(args.out_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics) + '\n')
                if wandb_run is not None:
                    wandb_run.log(metrics)

        # D. 保存检查点 (每 1000 步保存一次)
        if (it + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, it + 1, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
