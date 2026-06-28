# DCASE2025 立体声迁移测试

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_azimuth_only_ifan_adapted_all_classes_balanced_run20_bg_20260616_210435/checkpoints/best_doa_error.pt`
- manifest: `/home/cmm/icocnn/datasets/DCASE2025_Task3/locata_like_all_classes_devtest/manifest_all.csv`
- 立体声代理基线间距: `0.080 m`
- 前部排除窗口数: `5`

| Subset | Clips | Windows | DOA error (deg) | Folded Az RMSE | Raw Az MAE | Raw Az RMSE | Horizontal Assumption RMSAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 3111 | 61618 | 39.6789 | 42.1797 | 39.6871 | 42.1872 | 42.1873 |
| moving_single_source | 731 | 14137 | 37.0105 | 40.9041 | 37.0148 | 40.9102 | 40.9102 |
| static_single_source | 2380 | 47481 | 40.4985 | 42.5715 | 40.5079 | 42.5795 | 42.5795 |

## Tail Diagnostics

| Scope | p75 | p90 | p95 | >=45 deg | >=60 deg | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 56.8351 | 90.0884 | 108.9196 | 1050 | 727 | 313 |

## Angle Bins

| Target folded azimuth bin | Clips | DOA error (deg) | p90 | p95 | >=90 deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| [-90,-60) | 482 | 60.3242 | 123.0442 | 135.1068 | 141 |
| [-60,-30) | 519 | 40.5703 | 87.0792 | 95.2757 | 44 |
| [-30,0) | 537 | 26.8131 | 53.9613 | 63.1586 | 1 |
| [0,30) | 574 | 28.1216 | 56.0567 | 63.6555 | 0 |
| [30,60) | 542 | 34.2803 | 80.3566 | 94.4543 | 36 |
| [60,90] | 457 | 52.9287 | 113.2137 | 129.0379 | 91 |

说明:
- 官方 DCASE Task 3 的 `DOA error (deg)` 定义是在匹配成功的 true-positive 事件上计算的。
- 在我们这个筛过的单声源子集上，由于每一帧只有一个活跃声源、也没有单独的检测分支，这里的 `DOA error (deg)` 近似为 active frame 上的 folded azimuth 平均绝对误差。
- DCASE 标签的方位角被折叠到前方视野，因此这里以 folded azimuth 作为主要定位观察量。
- 由于 DCASE2025 Task3 不提供 elevation 标注，`Horizontal Assumption RMSAE` 将目标 elevation 固定为 `0 deg`。
- 这是一项基于 2 通道 M/S stereo 简单双耳间距代理前端的迁移测试，不等价于 LOCATA `benchmark2` 阵列评测。
