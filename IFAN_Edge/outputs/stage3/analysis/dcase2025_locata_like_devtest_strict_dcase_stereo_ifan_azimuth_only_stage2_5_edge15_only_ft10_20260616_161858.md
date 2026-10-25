# DCASE2025 立体声迁移测试

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_azimuth_only_stage2_5_edge15_only_ft10_20260616_161858/checkpoints/best_doa_error.pt`
- manifest: `/home/cmm/icocnn/datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv`
- 立体声代理基线间距: `0.080 m`
- 前部排除窗口数: `5`

| Subset | Clips | Windows | DOA error (deg) | Folded Az RMSE | Raw Az MAE | Raw Az RMSE | Horizontal Assumption RMSAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 727 | 14471 | 29.4023 | 30.8755 | 29.4023 | 30.8755 | 30.8755 |
| moving_single_source | 15 | 300 | 30.5004 | 32.6862 | 30.5004 | 32.6862 | 32.6862 |
| static_single_source | 712 | 14171 | 29.3792 | 30.8373 | 29.3792 | 30.8373 | 30.8374 |

## Tail Diagnostics

| Scope | p75 | p90 | p95 | >=45 deg | >=60 deg | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 35.2753 | 74.3951 | 103.6780 | 138 | 102 | 48 |

## Angle Bins

| Target folded azimuth bin | Clips | DOA error (deg) | p90 | p95 | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| [-90,-60) | 103 | 42.9793 | 119.4625 | 128.9495 | 16 |
| [-60,-30) | 124 | 26.8140 | 69.0159 | 80.3650 | 4 |
| [-30,0) | 123 | 19.8323 | 37.5726 | 48.2532 | 0 |
| [0,30) | 129 | 19.8640 | 42.7466 | 51.5155 | 0 |
| [30,60) | 130 | 28.8965 | 77.4067 | 88.7301 | 5 |
| [60,90] | 118 | 41.2317 | 116.0970 | 132.4641 | 23 |

说明:
- 官方 DCASE Task 3 的 `DOA error (deg)` 定义是在匹配成功的 true-positive 事件上计算的。
- 在我们这个筛过的单声源子集上，由于每一帧只有一个活跃声源、也没有单独的检测分支，这里的 `DOA error (deg)` 近似为 active frame 上的 folded azimuth 平均绝对误差。
- DCASE 标签的方位角被折叠到前方视野，因此这里以 folded azimuth 作为主要定位观察量。
- 由于 DCASE2025 Task3 不提供 elevation 标注，`Horizontal Assumption RMSAE` 将目标 elevation 固定为 `0 deg`。
- 这是一项基于 2 通道 M/S stereo 简单双耳间距代理前端的迁移测试，不等价于 LOCATA `benchmark2` 阵列评测。
