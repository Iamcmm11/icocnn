"""Write a factual Chinese report from machine-readable evidence, including failures."""
from pathlib import Path
import json
import math
HERE=Path(__file__).resolve().parent;A=HERE/"artifacts"
def load(name,default=None):
    p=A/name
    return json.loads(p.read_text()) if p.exists() else default
def main():
    numerical=load("numerical_summary.json",{})
    hardware=load("hardware_summary.json",[])
    selection=load("selection.json",{})
    names=["完整张量","位置分块","流水融合","位置交错","投影折叠"]
    lines=["# MABA HLS 实验报告","",
           "对象：现有 IFAN_Edge 读出前自定义 MABA；FP32；VC707 xc7vx485tffg1761-2；目标时钟 5 ns。",
           "本报告由已完成实验的 JSON 汇总生成。未生成通过报告的任务不视为通过。","",
           "## 数值与链路验证","",
           "| 配置 | 用例数 | 最大绝对误差 | 最坏单用例 RMSE | 通过 |",
           "|---|---:|---:|---:|---|"]
    for r in numerical.get("module",[]):
        lines.append(f"| {names[r['variant']]} | {r['cases']} | {r['max_abs']:.8g} | {r['rmse']:.8g} | {r['passed']} |")
    lines+=["","模块门限保持最大绝对误差 1e-5、RMSE 1e-6。覆盖场景窗口、尾帧、脉冲、零方差、门控饱和和重复调用。",
            "96 帧长序列结果见 long_sequence_error.json/PDF，状态仍按调用复位。",""]
    for mode in ["fp32","mixed"]:
        r=load(f"chain_v4_{mode}.json")
        if r:
            lines += [f"### {'浮点' if mode=='fp32' else '现有混合精度'}链路","",
                      "\n".join("    "+s for s in r["log"].splitlines() if s.startswith("CHAIN")),""]
    lines+=["增量通过只说明替换 MABA 的扰动满足门限；混合精度整链总误差的未通过状态仍保留。",
            "多场景方向比较使用独立 FP64 软件读出；近零方向向量（任一范数 ≤1e-6）的夹角记为未定义，不以噪声方向报告大角度误差。","",
            "## 硬件结果","",
            "| 配置 | 估计时钟 ns | HLS 最坏周期 | RTL 最大周期 | DSP | BRAM18K | LUT | FF | RTL通过 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in hardware:
        lines.append(f"| {r['tag']} | {r['clock_ns']} | {r['cycles']} | {r['rtl_cycles'] if r['rtl_cycles'] is not None else '未完成'} | {r['DSP']} | {r['BRAM_18K']} | {r['LUT']} | {r['FF']} | {r['rtl_pass']} |")
    lines+=["","顶层周期包含权重加载、布局转换、AXI 输入输出和计算；时钟为 HLS 估计，未经过布局布线。",
            "参数扫描与五项逐步消融分别看待，不将并行度提高带来的收益归因于算子融合。",""]
    chosen=selection.get("selected")
    if chosen:
        lines+=["## 最终候选","",f"按数值通过、RTL通过、时钟及器件容量约束筛选，当前选择：**{chosen['tag']}**。",
                "筛选先取资源/延迟 Pareto 集，再按 RTL 延迟、BRAM、DSP、LUT、FF 排序。",""]
    else:
        lines+=["## 当前验收状态","","**尚无满足全部筛选条件的最终候选。** 需完成相应 RTL 仿真或处理失败项。",""]
    base=next((r for r in hardware if r["tag"]=="v0_t16_p1_T6_f8"),None)
    fused=next((r for r in hardware if r["tag"]=="v2_t16_p1_T6_f8"),None)
    inter=next((r for r in hardware if r["tag"]=="v3_t16_p1_T6_f8"),None)
    if base and fused and base["cycles"] and fused["cycles"]:
        lines += [f"综合层面，流水融合相对完整张量的周期比为 {base['cycles']/fused['cycles']:.3f}，DSP 减少 {(1-fused['DSP']/base['DSP'])*100:.2f}%，BRAM18K 减少 {(1-fused['BRAM_18K']/base['BRAM_18K'])*100:.2f}%。"]
    if fused and inter and fused["cycles"] and inter["cycles"]:
        lines += [f"位置交错相对融合版的估计周期变化为 {(inter['cycles']/fused['cycles']-1)*100:.6f}%。若变化接近零，结论限定为当前流水瓶颈未被交错显著改善。"]
    lines += ["","## 复现与结论边界","",
              "README.md 给出命令，METHOD.md 给出推导、缓存公式及公开工作的引用关系；artifacts/references.json 固定参考仓库 commit。",
              "保留原始权重和检查点，不修改 V14 原稿。无新增训练、完整 LOCATA 精度结论、板级吞吐/功耗或整网 RTL 通过声明。",
              "工具中断或重跑记录不代表算法失败，也不计作完成；最后的通过状态以对应原始报告为准。"]
    (HERE/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Wrote "+str(HERE/"REPORT.md"))
if __name__=="__main__":main()
