"""验证训练数值、标签对齐、检查点恢复和真实训练入口。"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy
from cs336_basics.optimizer import AdamW, clip_gradient_norm
from cs336_basics.scheduler import get_lr_cosine_schedule


class TrainingTests(unittest.TestCase):
    def test_loss_and_gradient_match_torch(self):
        torch.manual_seed(0)
        x = (torch.randn(2, 3, 7, dtype=torch.float64) * 100).requires_grad_()
        y = torch.randint(7, (2, 3))
        actual = cross_entropy(x, y)
        expected = F.cross_entropy(x.flatten(0, 1), y.flatten())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0],
                                   torch.autograd.grad(expected, x)[0])
        self.assertAlmostEqual(cross_entropy(torch.full((1, 2), 1e8), torch.tensor([0])).item(),
                               np.log(2), places=6)

    def test_batch_alignment_and_minimum_length(self):
        x, y = get_batch(np.arange(5, dtype=np.uint16), 3, 4, 'cpu')
        torch.testing.assert_close(x, torch.tensor([[0, 1, 2, 3]]).repeat(3, 1))
        torch.testing.assert_close(y, x + 1)
        self.assertEqual(x.dtype, torch.long)
        with self.assertRaises(ValueError):
            get_batch(np.arange(4), 1, 4, 'cpu')

    def test_adamw_course_formula_and_no_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.tensor([3.]))
        opt = AdamW([p, unused], lr=.02, betas=(.8, .9), eps=1e-6, weight_decay=.1)
        expected = p.detach().clone()
        m, v = torch.zeros_like(p), torch.zeros_like(p)
        for t in range(1, 5):
            g = torch.tensor([t * .3, -.4], dtype=p.dtype)
            p.grad = g
            m = .8 * m + .2 * g
            v = .9 * v + .1 * g.square()
            expected = (expected - .02 * ((1 - .9**t)**.5 / (1 - .8**t)) * m / (v.sqrt() + 1e-6)) * .998
            opt.step()
            torch.testing.assert_close(p, expected)
        self.assertEqual(unused.item(), 3.)

    def test_global_clip_and_schedule(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([3., 4.])
        clip_gradient_norm(iter([p]), 2.)
        torch.testing.assert_close(p.grad, torch.tensor([1.2, 1.6]))
        values = [get_lr_cosine_schedule(t, 1., .1, 2, 6) for t in (0, 2, 4, 6, 7)]
        np.testing.assert_allclose(values, [0, 1, .55, .1, .1])
        self.assertEqual(get_lr_cosine_schedule(0, 1., .1, 0, 6), 1.)
        with self.assertRaises(ValueError):
            get_lr_cosine_schedule(0, 1., .1, 2, 2)

    def test_checkpoint_restores_next_update_and_rng(self):
        model = torch.nn.Linear(2, 1)
        opt = AdamW(model.parameters())
        def step():
            opt.zero_grad()
            model(torch.randn(3, 2)).square().mean().backward()
            opt.step()
        step()
        f = io.BytesIO()
        save_checkpoint(model, opt, 1, f)
        step()
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        f.seek(0)
        self.assertEqual(load_checkpoint(f, model, opt), 1)
        step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])

    def test_cpu_entrypoint_and_resume(self):
        from cs336_basics.main_train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'tokens.bin'
            np.tile(np.arange(8, dtype=np.uint16), 8).tofile(data)
            common = ['--train_data_path', str(data), '--valid_data_path', str(data),
                      '--device', 'cpu', '--vocab_size', '8', '--d_model', '8',
                      '--num_heads', '2', '--num_layers', '1', '--d_ff', '16',
                      '--context_length', '4', '--batch_size', '2', '--warmup_iters', '0',
                      '--eval_batches', '1', '--eval_interval', '1', '--save_interval', '1',
                      '--out_dir', str(root / 'out')]
            with contextlib.redirect_stdout(io.StringIO()):
                main(common + ['--max_iters', '2'])
                main(common + ['--max_iters', '3', '--resume', str(root / 'out/ckpt.pt')])
            checkpoint = torch.load(root / 'out/ckpt.pt', weights_only=True)
            self.assertEqual(checkpoint['iteration'], 3)
            self.assertEqual(len((root / 'out/metrics.jsonl').read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
