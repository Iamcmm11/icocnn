"""Training entrypoint for IcoTempCNN + lightweight MABA experiments."""

import argparse
import copy
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners
import acousticTrackingModels as at_models
from acousticTrackingDataset import Parameter

from maba_doa.models import IcoTempCNNWithMABA

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(cfg):
    ksize = int(cfg["data"]["window"]["K"])
    hop_ratio = float(cfg["data"]["window"]["hop_ratio"])
    trajectory_seconds = int(cfg["data"]["trajectory_seconds"])

    path_train = cfg["data"]["train_path"]
    path_test = cfg["data"]["test_path"]

    corpus_train = at_dataset.LibriSpeechDataset(path_train, trajectory_seconds, return_vad=True)
    corpus_test = at_dataset.LibriSpeechDataset(path_test, trajectory_seconds, return_vad=True)
    windowing = at_dataset.Windowing(ksize, int(ksize * hop_ratio), window=np.hanning)

    room_sz = Parameter([3, 3, 2.5], [10, 8, 6])
    t60 = Parameter(0.2, 1.3)
    abs_weights = Parameter([0.5] * 6, [1.0] * 6)
    array_setup = at_dataset.benchmark2_array_setup
    array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5])
    snr_start = float(cfg["data"]["snr"]["start"])
    nb_points = int(cfg["data"]["nb_points"])

    dataset_train = at_dataset.RandomTrajectoryDataset(
        sourceDataset=corpus_train,
        room_sz=room_sz,
        T60=t60,
        abs_weights=abs_weights,
        array_setup=array_setup,
        array_pos=array_pos,
        SNR=Parameter(snr_start),
        nb_points=nb_points,
        transforms=[windowing],
    )
    dataset_test = at_dataset.RandomTrajectoryDataset(
        sourceDataset=corpus_test,
        room_sz=room_sz,
        T60=t60,
        abs_weights=abs_weights,
        array_setup=array_setup,
        array_pos=array_pos,
        SNR=Parameter(snr_start),
        nb_points=nb_points,
        transforms=[windowing],
    )
    return dataset_train, dataset_test, array_setup


def build_model(cfg):
    r = int(cfg["data"]["r"])
    channels = int(cfg["model"]["channels"])
    cin = int(cfg["model"]["cin"])
    smooth_vertices = bool(cfg["model"]["smooth_vertices"])
    variant = cfg["model"]["variant"]

    if variant == "baseline":
        return at_models.IcoTempCNN(r, channels, Cin=cin, smooth_vertices=smooth_vertices)

    mcfg = cfg["model"]["maba"]
    use_gate = variant != "ablation_no_gate"
    use_state = variant != "ablation_no_state"
    return IcoTempCNNWithMABA(
        r=r,
        C=channels,
        Cin=cin,
        smooth_vertices=smooth_vertices,
        maba_d_model=int(mcfg["d_model"]),
        maba_state_dim=int(mcfg["state_dim"]),
        maba_conv_kernel=int(mcfg["conv_kernel"]),
        dropout=float(mcfg["dropout"]),
        use_residual=bool(mcfg["use_residual"]),
        use_gate=use_gate,
        use_state=use_state,
    )


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def benchmark_step_ms(model, x_batch, repeat=20):
    model.eval()
    use_cuda = x_batch.device.type == "cuda"
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_batch)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(x_batch)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / max(repeat, 1)


def prepare_output_dir(cfg, output_suffix=None):
    root = cfg["experiment"]["output_root"]
    name = cfg["experiment"]["name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_suffix:
        dirname = "{}_{}_{}".format(name, output_suffix, stamp)
    else:
        dirname = "{}_{}".format(name, stamp)
    out_dir = os.path.join(root, dirname)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_training(cfg, output_suffix=None):
    cfg = copy.deepcopy(cfg)
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    out_dir = prepare_output_dir(cfg, output_suffix=output_suffix)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    dataset_train, dataset_test, array_setup = build_datasets(cfg)
    model = build_model(cfg)
    fs = int(cfg["data"]["fs"])
    ksize = int(cfg["data"]["window"]["K"])
    preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
        N=array_setup.mic_pos.shape[0],
        K=ksize,
        r=int(cfg["data"]["r"]),
        rn=array_setup.mic_pos,
        fs=fs,
        apply_vad=bool(cfg["data"]["use_vad"]),
    )
    learner = at_learners.OneSourceTrackingLearner(model, preprocessor)

    requested_device = cfg.get("device", "cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        learner.cuda()
        device = "cuda"
    else:
        device = "cpu"

    epochs = int(cfg["training"]["epochs"])
    trajectories_per_batch = int(cfg["training"]["trajectories_per_batch"])
    trajectories_per_gpu_call = int(cfg["training"]["trajectories_per_gpu_call"])
    lr = float(cfg["training"]["lr"])

    schedule_cfg = cfg["training"].get("snr_schedule", {})
    schedule_enable = bool(schedule_cfg.get("enable", False))
    schedule_epoch = int(schedule_cfg.get("epoch", -1))
    schedule_min = float(schedule_cfg.get("min", 5.0))
    schedule_max = float(schedule_cfg.get("max", 30.0))

    history_rows = []
    for epoch_idx in range(epochs):
        t0 = time.perf_counter()
        learner.train_epoch(
            dataset_train,
            trajectories_per_batch=trajectories_per_batch,
            trajectories_per_gpu_call=trajectories_per_gpu_call,
            lr=lr,
            epoch=epoch_idx,
        )
        loss_test, rmsae_test = learner.test_epoch(
            dataset_test,
            trajectories_per_batch=int(cfg["evaluation"]["trajectories_per_batch"]),
        )
        elapsed = time.perf_counter() - t0

        history_rows.append(
            {
                "epoch": epoch_idx + 1,
                "test_loss": float(loss_test),
                "test_rmsae_deg": float(rmsae_test),
                "epoch_time_s": elapsed,
                "lr": lr,
            }
        )

        if schedule_enable and epoch_idx == schedule_epoch:
            dataset_train.SNR = Parameter(schedule_min, schedule_max)
            dataset_test.SNR = Parameter(schedule_min, schedule_max)

    model_path = os.path.join(out_dir, "model.bin")
    torch.save(model.state_dict(), model_path)

    # Latency and MAC proxy use one realistic sample from preprocessor output.
    mic_sig_batch, acoustic_scene_batch = dataset_test.get_batch(0, 1)
    x_batch, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
    latency_ms = benchmark_step_ms(
        model,
        x_batch,
        repeat=int(cfg["evaluation"]["latency_repeat"]),
    )

    maba_mac_proxy = 0
    if hasattr(model, "maba"):
        maba_mac_proxy = int(model.maba.mac_proxy(time_steps=x_batch.shape[2]))

    final_loss = history_rows[-1]["test_loss"] if history_rows else None
    final_rmsae = history_rows[-1]["test_rmsae_deg"] if history_rows else None
    summary = {
        "variant": cfg["model"]["variant"],
        "device": device,
        "seed": seed,
        "param_count": count_params(model),
        "maba_mac_proxy": maba_mac_proxy,
        "latency_step_ms": latency_ms,
        "final_test_loss": final_loss,
        "final_test_rmsae_deg": final_rmsae,
        "model_path": model_path,
        "output_dir": out_dir,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        if history_rows:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        else:
            f.write("epoch,test_loss,test_rmsae_deg,epoch_time_s,lr\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train IcoTempCNN + lightweight MABA DOA model.")
    parser.add_argument("--config", type=str, default="maba_doa/configs/default.yaml")
    parser.add_argument("--variant", type=str, default=None, choices=[
        "baseline",
        "maba",
        "ablation_no_gate",
        "ablation_no_state",
    ])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-suffix", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.variant is not None:
        cfg["model"]["variant"] = args.variant
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.cpu:
        cfg["device"] = "cpu"

    summary = run_training(cfg, output_suffix=args.output_suffix)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
