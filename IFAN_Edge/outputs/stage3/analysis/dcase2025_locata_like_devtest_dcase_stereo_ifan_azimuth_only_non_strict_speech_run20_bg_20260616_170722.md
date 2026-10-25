# DCASE2025 立体声迁移测试

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_azimuth_only_non_strict_speech_run20_bg_20260616_170722/checkpoints/best_doa_error.pt`
- manifest: `/home/cmm/icocnn/datasets/DCASE2025_Task3/locata_like_devtest/manifest_all.csv`
- 立体声代理基线间距: `0.080 m`
- 前部排除窗口数: `5`

| Subset | Clips | Windows | DOA error (deg) | Folded Az RMSE | Raw Az MAE | Raw Az RMSE | Horizontal Assumption RMSAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 1054 | 20620 | 32.2351 | 34.6414 | 32.2437 | 34.6478 | 34.6478 |
| moving_single_source | 340 | 6413 | 39.3882 | 43.9139 | 39.3965 | 43.9195 | 43.9195 |
| static_single_source | 714 | 14207 | 28.8288 | 30.2260 | 28.8376 | 30.2327 | 30.2328 |

## Tail Diagnostics

| Scope | p75 | p90 | p95 | >=45 deg | >=60 deg | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 42.6755 | 75.6838 | 98.5695 | 247 | 161 | 71 |

## Angle Bins

| Target folded azimuth bin | Clips | DOA error (deg) | p90 | p95 | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| [-90,-60) | 140 | 47.5989 | 110.1171 | 131.0921 | 23 |
| [-60,-30) | 178 | 32.6358 | 72.8988 | 84.3945 | 7 |
| [-30,0) | 188 | 24.3429 | 49.7191 | 57.7554 | 1 |
| [0,30) | 202 | 21.9873 | 45.1669 | 57.9117 | 0 |
| [30,60) | 193 | 32.9406 | 83.9592 | 98.8565 | 16 |
| [60,90] | 153 | 40.0478 | 109.9182 | 125.5936 | 24 |

说明:
- 官方 DCASE Task 3 的 `DOA error (deg)` 定义是在匹配成功的 true-positive 事件上计算的。
- 在我们这个筛过的单声源子集上，由于每一帧只有一个活跃声源、也没有单独的检测分支，这里的 `DOA error (deg)` 近似为 active frame 上的 folded azimuth 平均绝对误差。
- DCASE 标签的方位角被折叠到前方视野，因此这里以 folded azimuth 作为主要定位观察量。
- 由于 DCASE2025 Task3 不提供 elevation 标注，`Horizontal Assumption RMSAE` 将目标 elevation 固定为 `0 deg`。
- 这是一项基于 2 通道 M/S stereo 简单双耳间距代理前端的迁移测试，不等价于 LOCATA `benchmark2` 阵列评测。
