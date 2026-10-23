"""Optional CPU tensor checks; no pretrained downloads or CUDA training."""
import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch is not installed")
class ClassifierTensorContract(unittest.TestCase):
    def test_last_nonpadding_state_and_head_roundtrip(self):
        import torch
        from hastika.models.llm_classifier import LLMClassifier
        class Backbone(torch.nn.Module):
            config = SimpleNamespace(hidden_size=2)
            def forward(self, input_ids, attention_mask):
                x = input_ids.float()
                return SimpleNamespace(last_hidden_state=torch.stack([x, x * 2], dim=-1))
        model = LLMClassifier(Backbone(), 2)
        model.head.load_state_dict({"weight": torch.eye(2), "bias": torch.zeros(2)})
        model.eval()
        x = torch.tensor([[2, 3, 0], [4, 0, 0]])
        m = x.ne(0).long()
        torch.testing.assert_close(model(x, m), torch.tensor([[3., 6.], [4., 8.]]))
        restored = LLMClassifier(Backbone(), 2)
        restored.head.load_state_dict(model.head.state_dict())
        restored.eval()
        torch.testing.assert_close(restored(x, m), model(x, m))
