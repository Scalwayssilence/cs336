"""用 CPU 小模型验证因果性、训练闭环和位置/门控的关键性质。"""

import math
import unittest

import torch
import torch.nn.functional as F

from cs336_basics.nn import (
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerLM,
    scaled_dot_product_attention,
)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def model(self):
        return TransformerLM(
            vocab_size=13, context_length=8, d_model=8,
            num_layers=2, num_heads=2, d_ff=16, rope_theta=10000.0,
            device="cpu", dtype=torch.float32,
        )

    def test_future_tokens_do_not_change_prefix_logits(self):
        model = self.model().eval()
        original = torch.tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
        changed = original.clone()
        changed[:, 3:] = torch.tensor([[11, 12], [0, 1]])
        with torch.no_grad():
            logits = model(original)
            changed_logits = model(changed)
            prefix_logits = model(original[:, :3])
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])
        torch.testing.assert_close(logits[:, :3], prefix_logits)
        self.assertFalse(torch.allclose(logits[:, 3:], changed_logits[:, 3:]))

    def test_cross_entropy_backward_and_optimizer_have_distinct_jobs(self):
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # 同一段文本切出输入和右移标签；每个位置预测其下一个 token。
        tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
        inputs, labels = tokens[:, :-1], tokens[:, 1:]
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.zero_grad()
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 4, 13))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss = F.cross_entropy(logits.reshape(-1, 13), labels.reshape(-1))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter, saved in zip(model.parameters(), before):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertTrue(torch.equal(parameter.detach(), saved))
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0
                            for parameter in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(parameter.detach(), saved)
                            for parameter, saved in zip(model.parameters(), before)))

    def test_attention_matches_known_weights_and_blocks_masked_gradients(self):
        # 缩放后两个有效分数为 1 和 0；第三个 Key 再大也应被屏蔽。
        q = torch.tensor([[math.sqrt(2.0), 0.0]], requires_grad=True)
        k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
                         requires_grad=True)
        v = torch.tensor([[2.0, -1.0], [4.0, 3.0], [1000.0, 1000.0]],
                         requires_grad=True)
        actual = scaled_dot_product_attention(q, k, v,
                                              torch.tensor([[True, True, False]]))
        first_weight = math.e / (math.e + 1.0)
        expected = torch.tensor([[2 * first_weight + 4 * (1 - first_weight),
                                  -first_weight + 3 * (1 - first_weight)]])
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.equal(k.grad[2], torch.zeros(2)))
        self.assertTrue(torch.equal(v.grad[2], torch.zeros(2)))
        self.assertGreater(k.grad[:2].abs().sum().item(), 0)
        self.assertGreater(v.grad[:2].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(q.grad).all().item())

    def test_rope_preserves_norm_and_depends_on_relative_position(self):
        rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, context_length=8)
        q, k = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4)
        at = lambda tensor, position: rope(tensor, torch.tensor([[position]]))
        q_at_2, k_at_5 = at(q, 2), at(k, 5)
        torch.testing.assert_close(q_at_2.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(k_at_5.norm(dim=-1), k.norm(dim=-1))
        # (R_2 q)·(R_5 k) = q·(R_3 k)，同时平移两个位置也保持该点积。
        dot = (q_at_2 * k_at_5).sum(dim=-1)
        torch.testing.assert_close(dot, (q * at(k, 3)).sum(dim=-1))
        torch.testing.assert_close(dot, (at(q, 3) * at(k, 6)).sum(dim=-1))

    def test_swiglu_does_not_mix_token_positions(self):
        ffn = SwiGLU(d_model=8, d_ff=16, device="cpu", dtype=torch.float32)
        inputs = torch.randn(2, 4, 8)
        changed = inputs.clone()
        changed[:, 2] += 5.0
        output, changed_output = ffn(inputs), ffn(changed)
        self.assertEqual(tuple(output.shape), (2, 4, 8))
        torch.testing.assert_close(output[:, [0, 1, 3]],
                                   changed_output[:, [0, 1, 3]])
        torch.testing.assert_close(output[:, 1], ffn(inputs[:, 1]))
        self.assertFalse(torch.allclose(output[:, 2], changed_output[:, 2]))


if __name__ == "__main__":
    unittest.main()
