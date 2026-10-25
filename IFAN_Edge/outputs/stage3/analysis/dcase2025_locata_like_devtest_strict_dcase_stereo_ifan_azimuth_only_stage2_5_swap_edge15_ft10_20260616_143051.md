# DCASE2025 立体声迁移测试

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_azimuth_only_stage2_5_swap_edge15_ft10_20260616_143051/checkpoints/best_doa_error.pt`
- manifest: `/home/cmm/icocnn/datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv`
- 立体声代理基线间距: `0.080 m`
- 前部排除窗口数: `5`

| Subset | Clips | Windows | DOA error (deg) | Folded Az RMSE | Raw Az MAE | Raw Az RMSE | Horizontal Assumption RMSAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 727 | 14471 | 29.4908 | 30.8253 | 29.4908 | 30.8253 | 30.8254 |
| moving_single_source | 15 | 300 | 30.4711 | 32.4794 | 30.4711 | 32.4794 | 32.4794 |
| static_single_source | 712 | 14171 | 29.4701 | 30.7905 | 29.4701 | 30.7905 | 30.7905 |

## Tail Diagnostics

| Scope | p75 | p90 | p95 | >=45 deg | >=60 deg | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 35.8971 | 71.9960 | 101.2073 | 138 | 96 | 47 |

## Angle Bins

| Target folded azimuth bin | Clips | DOA error (deg) | p90 | p95 | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| [-90,-60) | 103 | 43.2578 | 112.2568 | 123.4326 | 15 |
| [-60,-30) | 124 | 26.8618 | 65.5395 | 76.1477 | 3 |
| [-30,0) | 123 | 19.5321 | 34.6763 | 44.6866 | 0 |
| [0,30) | 129 | 17.6963 | 38.6427 | 47.9520 | 0 |
| [30,60) | 130 | 29.1436 | 77.6718 | 86.5754 | 5 |
| [60,90] | 118 | 43.8935 | 114.0317 | 130.0031 | 24 |

说明:
- 官方 DCASE Task 3 的 `DOA error (deg)` 定义是在匹配成功的 true-positive 事件上计算的。
- 在我们这个筛过的单声源子集上，由于每一帧只有一个活跃声源、也没有单独的检测分支，这里的 `DOA error (deg)` 近似为 active frame 上的 folded azimuth 平均绝对误差。
- DCASE 标签的方位角被折叠到前方视野，因此这里以 folded azimuth 作为主要定位观察量。
- 由于 DCASE2025 Task3 不提供 elevation 标注，`Horizontal Assumption RMSAE` 将目标 elevation 固定为 `0 deg`。
- 这是一项基于 2 通道 M/S stereo 简单双耳间距代理前端的迁移测试，不等价于 LOCATA `benchmark2` 阵列评测。
