import unittest

import torch

from maba_doa.models import IcoTempCNNWithMABA, MABATemporalRefiner


class TestMABATemporalRefiner(unittest.TestCase):
    def test_shape_and_grad(self):
        model = MABATemporalRefiner(
            charts=5,
            height=2,
            width=4,
            d_model=32,
            state_dim=8,
            conv_kernel=3,
            dropout=0.0,
            use_residual=True,
            use_gate=True,
            use_state=True,
        )
        x = torch.randn(2, 6, 5, 2, 4, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        loss = y.square().mean()
        loss.backward()
        grad_ok = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(grad_ok)

    def test_no_nan_inf(self):
        model = MABATemporalRefiner(dropout=0.0)
        x = torch.randn(1, 4, 5, 2, 4)
        y = model(x)
        self.assertFalse(torch.isnan(y).any().item())
        self.assertFalse(torch.isinf(y).any().item())


class TestIcoTempCNNWithMABA(unittest.TestCase):
    def test_forward_and_return_maps(self):
        model = IcoTempCNNWithMABA(
            r=2,
            C=32,
            Cin=1,
            smooth_vertices=True,
            maba_d_model=32,
            maba_state_dim=8,
            maba_conv_kernel=3,
            dropout=0.0,
            use_residual=True,
            use_gate=True,
            use_state=True,
        )
        x = torch.randn(1, 1, 8, 5, 4, 8)
        y = model(x)
        self.assertEqual(y.shape[-1], 3)

        y2, maps = model(x, return_maps=True)
        self.assertEqual(tuple(y2.shape), tuple(y.shape))
        self.assertIn("maps_before", maps)
        self.assertIn("maps_after", maps)
        self.assertIn("maps_clean", maps)
        self.assertEqual(maps["maps_before"].shape, maps["maps_after"].shape)


if __name__ == "__main__":
    unittest.main()
