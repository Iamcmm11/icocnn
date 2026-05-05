# Vitis HLS 神经网络优化指令速查（面向完整映射方案）

基于 `UG1399 v2025.2`（2026-01-22），整理适合神经网络算子（卷积/矩阵乘/逐点）的一组高价值优化项。  
目标：先在“完整映射”基础上加 pragma/directive，观察资源与吞吐变化。

## 1. 数据级优化（Data-Level）

| 方法 | 常用指令/类型 | 关键参数 | 典型效果 | 代价/注意 |
|---|---|---|---|---|
| SIMD 向量化 | `hls::vector<T,N>` | `T`,`N`（建议 2 的幂） | 单周期并行处理 N 路数据 | 需匹配存储带宽；可能增加端口和布线 |
| 数组分块并行读写 | `#pragma HLS array_partition` | `type=complete/block/cyclic`, `factor`, `dim` | 增加并行访问端口，支撑更低 II | BRAM/LUT/寄存器显著增加 |
| 位宽打包 | `#pragma HLS array_reshape` | `type`, `factor`, `dim` | 提高单次读写位宽，降低访存次数 | 可能增加宽总线和拼接逻辑 |
| 接口自动加宽 | `#pragma HLS interface mode=m_axi` | `max_widen_bitwidth` | 提升 AXI 吞吐 | 上游/下游位宽需匹配 |

提示：`array_partition/array_reshape` 不支持顶层 `m_axi` 端口，顶层宽并行更适合用 `hls::vector`。

## 2. 定点量化与数值类型优化

| 方法 | 关键语法 | 可选参数 | 典型效果 | 注意 |
|---|---|---|---|---|
| 整数位宽裁剪 | `ap_int<W> / ap_uint<W>` | `W` | DSP/LUT/时序明显改善 | 边界溢出要验证 |
| 定点替代浮点 | `ap_fixed<W,I,Q,O,N>` / `ap_ufixed<>` | `W`,`I`,`Q`,`O`,`N` | 显著降 DSP 与延迟 | 精度需离线量化评估 |
| 量化策略 | `Q` | `AP_TRN`(默认), `AP_RND`, `AP_RND_ZERO`... | 控制舍入误差 | 更复杂舍入会增逻辑 |
| 溢出策略 | `O` | `AP_WRAP`(默认), `AP_SAT`, `AP_SAT_SYM`... | 控制溢出行为 | `AP_SAT*` 常增加 LUT（文档提示最高可到约 20%） |

实用建议：先用 `AP_TRN + AP_WRAP` 做资源基线，再按精度需求逐步切到 `AP_RND/AP_SAT`。

## 3. 流水线与数据流优化

| 方法 | 指令 | 关键参数 | 作用 |
|---|---|---|---|
| 循环/函数流水 | `#pragma HLS pipeline` | `II`, `rewind`, `style=stp/flp/frp` | 降低启动间隔，提升吞吐 |
| 任务级并行 | `#pragma HLS dataflow` | `disable_start_propagation` | 子函数/子循环重叠执行 |
| 通道化 | `#pragma HLS stream` | `type=fifo/pipo/shared/unsync`, `depth` | 用通道替代 RAM ping-pong，降低等待 |
| 依赖打破 | `#pragma HLS dependence` | `type=inter/intra`, `direction=RAW/WAR/WAW`, `distance`, `true/false` | 去除伪相关，帮助 `II=1` |
| 稳定输入声明 | `#pragma HLS stable` | `variable` | 减少 dataflow 同步负担，常可降 II |

补充：
- `pipeline style`：`stp` 默认；`frp` 常用于减控制扇出/减死锁风险；`flp` 可冲刷流水但可能更耗资源。
- `loop_tripcount` 只影响分析报表，不改变综合结果。

## 4. 资源复用与共享优化

| 方法 | 指令 | 关键参数 | 典型用途 |
|---|---|---|---|
| 限制实例数 | `#pragma HLS allocation` | `type=function/operation`, `instances`, `limit` | 控 DSP/LUT，上吞吐换面积 |
| 运算绑定 | `#pragma HLS bind_op` | `op`, `impl=fabric/dsp/...`, `latency` | 指定乘加映射到 DSP 或 LUT |
| 存储绑定 | `#pragma HLS bind_storage` | `type=RAM_1P/RAM_2P/FIFO/...`, `impl=bram/uram/lutram/...`, `latency` | 精细控制 BRAM/URAM/LUTRAM |
| 函数实例化 | `#pragma HLS function_instantiate` | `variable` | 按常量参数生成专用实例 |
| 内联控制 | `#pragma HLS inline` | `off`, `recursive` | 在“共享硬件”和“打平优化”间权衡 |

## 5. 存储层次与接口优化

| 层次 | 主要参数/指令 | 作用 | 注意 |
|---|---|---|---|
| 片外 AXI | `interface m_axi` 的 `max_read_burst_length`, `max_write_burst_length`, `num_read_outstanding`, `num_write_outstanding`, `latency` | 提升 DDR/HBM 带宽利用率 | outstanding 增大时会引入更大内部 FIFO |
| 片外读缓存 | `#pragma HLS cache` | `lines`, `depth`（及 L2 相关） | 非 burst/邻近访问场景降低平均访存延迟 | 会增加片上资源 |
| 片上存储绑定 | `bind_storage` + `array_partition/reshape` | 多端口与层次化存储 | 需和并行度匹配，否则端口仍瓶颈 |
| 接口分组 | `interface bundle`/`channel` | 合理拆分 AXI 通道 | bundle 命名建议小写 |

## 6. 针对你当前“完整映射”方案的优先尝试顺序

1. **先做吞吐主线**：对最内层 MAC 循环加 `pipeline II=1`，再用 `dependence ... false` 消伪相关。  
2. **补带宽**：对热点数组做 `array_partition`（优先 `dim` 在并行读维度），必要时 `array_reshape`。  
3. **做阶段并行**：把 `pad -> conv -> bias/激活 -> reorder` 放进 `dataflow`，中间通道用 `stream depth=2~32` 扫描。  
4. **压资源**：通过 `allocation limit` 和 `bind_op/bind_storage` 做“并行度-资源”折中。  
5. **最后做数值优化**：把 `float` 切到 `ap_fixed`（先 `TRN+WRAP`），用 C 仿真/对比脚本验证精度后再收紧位宽。

## 7. 建议的最小参数扫描集合（第一轮）

- `pipeline II`: `1 / 2`
- `unroll factor`（关键维度）: `2 / 4 / 8`
- `array_partition factor`: `2 / 4 / 8`（按读并发需求）
- `stream depth`: `2 / 4 / 8 / 16`
- `allocation limit(mul)`: `full / 1/2 / 1/4`
- `data_t`: `float` -> `ap_fixed<16,6>` -> `ap_fixed<12,4>`

---

参考章节（UG1399 v2025.2）：
- Chapter 17 `HLS Pragmas`（`pipeline/dataflow/stream/unroll/interface/array_partition/bind_*` 等）
- Chapter 18 `HLS Tcl Commands`（`set_directive_*`, `config_*`）
- Chapter 6 + Chapter 21（`ap_int/ap_fixed/hls::vector` 与数值行为）
