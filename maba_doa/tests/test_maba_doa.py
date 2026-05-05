import unittest
import json
import os
import runpy
import tempfile
from pathlib import Path

import torch

from maba_doa.models import (
    IcoTempCNNReplaceTemporalMABA,
    IcoTempCNNWithMABA,
    MABAChannelTemporalBlock,
    MABATemporalRefiner,
)


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


class TestMABAChannelTemporalBlock(unittest.TestCase):
    def test_shape_and_grad(self):
        model = MABAChannelTemporalBlock(
            channels=32,
            d_model=32,
            state_dim=8,
            conv_kernel=3,
            dropout=0.0,
            use_residual=True,
            use_gate=True,
            use_state=True,
        )
        x = torch.randn(4, 32, 7, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        y.square().mean().backward()
        grad_ok = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(grad_ok)

    def test_causality_smoke(self):
        model = MABAChannelTemporalBlock(
            channels=4,
            d_model=4,
            state_dim=2,
            conv_kernel=3,
            dropout=0.0,
            use_residual=True,
            use_gate=True,
            use_state=True,
        )
        model.eval()
        x1 = torch.zeros(1, 4, 6)
        x2 = x1.clone()
        x2[:, :, -1] = 1.0
        y1 = model(x1)
        y2 = model(x2)
        self.assertTrue(torch.allclose(y1[:, :, :-1], y2[:, :, :-1], atol=1e-6, rtol=1e-5))


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


class TestIcoTempCNNReplaceTemporalMABA(unittest.TestCase):
    def test_forward_shape(self):
        model = IcoTempCNNReplaceTemporalMABA(
            r=2,
            C=32,
            Cin=1,
            smooth_vertices=True,
            replace_d_model=32,
            replace_state_dim=8,
            replace_conv_kernel=3,
            dropout=0.0,
            use_residual=True,
        )
        x = torch.randn(1, 1, 8, 5, 4, 8)
        y = model(x)
        self.assertEqual(y.shape[-1], 3)
        self.assertIsInstance(model.temp_cnn[0], MABAChannelTemporalBlock)
        self.assertEqual(model.temp_cnn[-1].conv.out_channels, 1)


class TestEvaluateLocataScript(unittest.TestCase):
    def setUp(self):
        module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "evaluate_locata.py"))
        self.normalize_tasks = module["normalize_tasks"]
        self.resolve_model_spec = module["resolve_model_spec"]
        self.build_report = module["build_report"]
        self.markdown_summary = module["markdown_summary"]

    def test_normalize_tasks_accepts_single_source_tasks(self):
        self.assertEqual(self.normalize_tasks([1, 3, 5]), (1, 3, 5))

    def test_normalize_tasks_rejects_other_tasks(self):
        with self.assertRaises(ValueError):
            self.normalize_tasks([2])

    def test_resolve_model_spec_from_summary_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            model_path = run_dir / "model.bin"
            model_path.write_bytes(b"")
            summary_path = run_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "variant": "maba",
                        "model_path": str(model_path),
                    },
                    f,
                )
            spec = self.resolve_model_spec(role="maba", summary_arg=str(run_dir), checkpoint_arg=None)
            self.assertEqual(spec["variant"], "maba")
            self.assertEqual(Path(spec["summary_path"]), summary_path.resolve())
            self.assertEqual(Path(spec["checkpoint_path"]), model_path.resolve())

    def test_build_report_and_markdown_contains_expected_sections(self):
        baseline_rows = [
            {
                "task": 1,
                "recording": "recording1",
                "array": "benchmark2",
                "directory": "/tmp/task1/recording1/benchmark2",
                "with_silences_rmsae_deg": 10.0,
                "without_silences_rmsae_deg": 8.0,
            }
        ]
        maba_rows = [
            {
                "task": 1,
                "recording": "recording1",
                "array": "benchmark2",
                "directory": "/tmp/task1/recording1/benchmark2",
                "with_silences_rmsae_deg": 9.0,
                "without_silences_rmsae_deg": 7.5,
            }
        ]
        report = self.build_report(
            baseline_spec={
                "variant": "baseline",
                "summary_path": "/tmp/baseline_summary.json",
                "checkpoint_path": "/tmp/baseline_model.bin",
            },
            maba_spec={
                "variant": "maba",
                "summary_path": "/tmp/maba_summary.json",
                "checkpoint_path": "/tmp/maba_model.bin",
            },
            baseline_rows=baseline_rows,
            maba_rows=maba_rows,
            config_path="/tmp/config.yaml",
            locata_root="/tmp/locata/dev",
            array="benchmark2",
            tasks=(1, 3, 5),
        )
        self.assertIn("recordings", report)
        self.assertIn("task_summary", report)
        self.assertIn("overall_summary", report)
        self.assertIn("comparison", report)
        md = self.markdown_summary(report)
        self.assertIn("baseline", md)
        self.assertIn("maba", md)
        self.assertIn("with silences", md.lower())
        self.assertIn("without silences", md.lower())
        self.assertIn("task1", md)


if __name__ == "__main__":
    unittest.main()
