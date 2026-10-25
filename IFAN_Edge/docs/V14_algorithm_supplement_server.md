# V14 算法补充实验：服务器执行说明

日期：2026-10-25。用途：补齐论文第三章“为什么采用规则压缩、为什么采用读出前 MABA”的实验依据。本说明只安排算法、软件推理及软件数值验证，不要求在服务器运行 FPGA/HLS。

**执行顺序：先 P0 复评已有检查点，回看结果，再安排 P1 训练；不要一开始启动全部扫描。** 本次 Git 提交仅增加本说明，下面标为“需补实现”的功能尚未随文档实现。命令按当前仓库脚本核对，服务器执行前仍须检查分支和 `--help`。

## 1. 先跑什么

| 优先级 | 实验 | 目的 | 现有入口 | 需要训练 |
|---|---|---|---|---|
| P0，立即跑 | 8 个已有候选的统一 LOCATA + 扩大模拟评估 | 排除基线混用、查明 2/8 与 3/8 的真实数据表现 | 可以运行，先完成严格加载预检 | 否 |
| P1-A，必须补 | C16 dense 继续训练与 4/8、3/8、2/8 同父模型微调对照；C8 同预算对照 | 区分稀疏损伤与训练预算影响 | CLI 已支持 | 是 |
| P1-B，必须补 | C8 无 MABA、读出后、读出前、双位置、读出前无状态/无动态门控 | 解释时序位置和状态机制的收益 | CLI 已支持 | 是 |
| P1-C，必须补 | PHAT-only、LMS-only、PHAT+LMS | 验证双特征互补性 | CLI 已支持输入置零消融 | 是 |
| P2-A，建议补 | C=4/8/12/16 宽度；C8 的 r2/r3 | 解释为何选 C8/r2 | 已有宽度 CLI；r3 配置已存在 | 是 |
| P2-B，有贡献声明时必须补 | 移除残差、固定融合、共享/独立注意力 | 证明这些组件在本文网络中的单独作用 | **需补模型/配置实现** | 是 |
| P2-C，机制分析 | 逐帧误差、转向片段、静音恢复、同录音轨迹图 | 解释 Task5 改善与 Task3 回退 | **需补逐帧导出/分析** | 不一定 |
| P2-D，部署衔接 | CPU/GPU 推理成本、整段与分块状态一致性 | 验证复杂度是否转化为执行收益及流式语义 | 性能入口已有；流式状态接口**需补实现** | 否 |

算力有限时，最低交付是：**P0 全部复评 + P1-A 的 seed42 + P1-B 的 seed42**。这只够形成初步结论。正式论文的关键对照应完成训练种子 42、43、44，不能把三个模拟测试种子当成三次训练。

## 2. 模型命名与现有证据

- **icoCNN**：算法比较基线，权重 `models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin`。
- **C16 dense / IFAN_80**：同族宽网络对照，历史参数约 125.457 K。
- **C8 dense / DFA-IcoNet-Edge**：规则通道收缩，无 MABA，约 31.561 K。
- **C8 + pre-readout MABA / DFA-IcoNet-Edge-MABA**：同一个部署候选，约 32.353 K，**不要作为两组重复实验**。
- **SAF-lite N/8**：C16 ConvIco 分组连接掩码，不是直接删去统一的全局输出通道。C16 张量形状不变；零权重不自动减少稠密内核执行时间。
- V13 的 **500.129 K DFA-IcoNet** 与 C16 125.457 K 不是同一条记录。未找到对应 checkpoint 前，不能把两者改名合并。
- 2026-05-18 的原读出前 MABA 与 2026-05-26 的重训练（约 32.321 K）是不同版本。后者可作额外复现对照，但不可覆盖前者。

目前发现的关键问题：

1. 原论文手工表的 icoCNN 均值 8.5718°/7.1976°，不同于多份原始 JSON 中一致的 23 段配对基线 7.738667°/7.120051°。所有新百分比必须从**同一批复评输出**计算，不能拼接这两组基线。
2. 2/8、3/8、4/8 均有 20 轮微调记录，只有 4/8 有现成完整 LOCATA JSON。最应补的是 2/8 和 3/8，但所有候选都要在同一环境复评。
3. 历史四场景均值：4/8≈10.397°，3/8≈11.434°，2/8≈13.151°；C8≈9.979°。不过 4/8 在最困难模拟场景与 LOCATA 去静音均值上有其优势，不能预先写成“所有剪枝都失败”。
4. 读出后/双位置在部分 LOCATA 总体指标优于读出前。需要同时给出困难场景、计算量和任务级表现，不能只挑有利于读出前的结果。

以上是安排补实验的依据，**不是要求新实验复现某个预定数值或得出某个预定结论**。

## 3. 统一实验协议

### 3.1 数据、训练和评价分开冻结

| 项目 | 要求 |
|---|---|
| LOCATA | `eval / benchmark2 / Task1,3,5`；13/5/5 段，共 23 段，不静默漏录音 |
| 采样/窗口 | 16 kHz；4096 点窗口；3072 点 hop；模拟轨迹20 s |
| 模拟场景 | S1=30 dB/T60=0.2 s；S2=30 dB/0.8 s；S3=5 dB/0.8 s；S4=5 dB/1.4 s |
| 模拟最终评估 | 每场景64条/测试种子；测试种子142、143、144；困难均值为S3/S4等权平均 |
| 模拟验证 | 缓存128条；仅用于选 checkpoint；与最终测试语音互斥 |
| 训练种子 | 42、43、44；各自独立训练/微调并报告结果 |
| 新训练数据 | train-clean-100训练；dev-clean选择checkpoint；test-clean只作冻结后最终模拟评估 |
| 定位指标 | 含静音/去静音 RMSAE；任务级及逐录音值；单位度 |
| 去静音 | 窗口VAD均值≥2/3；保存掩码规则，不加预测幅值筛选 |
| 启动帧 | 当前模拟指标超过5帧时剔除前5帧，LOCATA默认不剔除；分别记录，不跨协议混算 |
| 角误差实现 | 当前 `utils.angular_error` 裁剪余弦至[-0.99999,0.99999]；所有模型一致，勿单独修改 |
| 状态 | 每段录音起点复位；同一录音整段forward；不把六帧分块复位结果混入整段评测 |
| 参数/MAC | 参数只计模型参数，不把state_dict的几何buffer当可训练参数；MAC固定[B=1,2,T=6,5,4,8]，r3另注明输入 |

**两种统计不能混用：**先对每段有效帧计算 RMSAE，再对23个录音值算术平均；任务均值也按录音聚合。录音间标准差描述场景差异，训练种子间均值±标准差描述训练不确定性。不要对3个任务均值计算标准差后标作23段录音统计；不要对各模型的std相减后当作配对差值的std。

**历史复评与新实验分表：**P0保留历史训练事实（过去使用test-clean选验证的限制仍然存在）；新P1采用dev-clean验证。换一套评估脚本不能消除历史数据使用问题。

### 3.2 当前脚本的几个实际限制

1. `evaluate_stage3_locata.py` 已有 `map_refiner.* → feature_refiner.*` 的读出前旧键名兼容；`evaluate_stage3_simulated.py` 当前直接strict加载，未做同样映射。下文先生成**仅作评估的兼容副本**，不删除归一化或其他参数。
2. `--input-ablation-mode phat_only/lms_only` 是把另一张特征图置零，仍保留双分支与前端计算。它检验信息贡献，**不是删除一个分支后的轻量模型**。训练和测试必须用相同置零方式；只在完整模型测试时置零是另外的失配实验。
3. `--map-maba-no-state` 用当前候选状态q代替递推；`--map-maba-no-gate` 用可学习的静态alpha代替动态门控。它们不是把整个模块删除。部分仍被构造的参数可能不参与输出，报告实际参数/执行路径，不能套用原MABA计数。
4. 启动训练的 `set_seed` 当前只显式设置NumPy/Torch；不能因seed相同就保证Python random、数据顺序及模拟输入逐样本相同。正式等预算对照须补Python random种子、固定语音/房间/轨迹清单及验证缓存，或保存生成清单哈希；按模型大小改变Torch随机消耗时尤其要检查。
5. 训练器当前只有 `test_split` 一个验证来源字段；训练中的四场景记录也使用这一来源。新训练将它设为dev-clean后，日志内四场景只能算开发集结果。最终模拟测试要用test-clean的**评估副本**，不能把开发集结果写成测试。
6. r3运行会改变其内置icoCNN前端的r参数。做r2/r3候选比较时，不能把r3报告中的baseline当固定r2 icoCNN；比较候选本身，并单独保留固定r2基线。

## 4. P0：已有模型统一复评（先执行这部分）

### 4.1 服务器准备和留档

以下为Linux Bash，历史服务器路径是/home/cmm/icocnn；若当前路径不同，先修改cd。使用已有可运行的算法环境，不在同一环境中随意升级Torch/gpuRIR。

~~~bash
cd /home/cmm/icocnn
conda activate icocnn
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42
export V14_RUN_ROOT="IFAN_Edge/outputs/algorithm_supplement_$(date +%Y%m%d_%H%M%S)"
export LOCATA_ROOT="datasets/LOCATA/LOCATA"
export LIBRISPEECH_ROOT="datasets/LibriSpeech"
mkdir -p "$V14_RUN_ROOT"/logs "$V14_RUN_ROOT"/locata "$V14_RUN_ROOT"/simulated "$V14_RUN_ROOT"/checkpoints_eval
git rev-parse HEAD > "$V14_RUN_ROOT/git_commit.txt"
git status --short > "$V14_RUN_ROOT/git_status.txt"
git diff --binary > "$V14_RUN_ROOT/worktree.patch"
python -m pip freeze > "$V14_RUN_ROOT/pip_freeze.txt"
nvidia-smi > "$V14_RUN_ROOT/nvidia_smi.txt"
python IFAN_Edge/scripts/train_stage3_ifan.py --help > "$V14_RUN_ROOT/train_help.txt"
python IFAN_Edge/scripts/evaluate_stage3_locata.py --help > "$V14_RUN_ROOT/locata_help.txt"
python IFAN_Edge/scripts/evaluate_stage3_simulated.py --help > "$V14_RUN_ROOT/simulated_help.txt"
~~~

先检查数据和依赖；任一条件不满足，停止正式实验，不用更少录音或随机数据代替。

~~~bash
python - <<'PY'
import os, json
from pathlib import Path
import torch, numpy, scipy, gpuRIR, webrtcvad, einops, tomli, soundfile
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'gpuRIR环境需要可用CUDA；不要悄悄回退CPU'
root = Path(os.environ['LOCATA_ROOT']) / 'eval'
records = {str(t): sorted(str(p) for p in (root / ('task'+str(t))).glob('recording*/benchmark2') if p.is_dir()) for t in (1,3,5)}
assert {k:len(v) for k,v in records.items()} == {'1':13, '3':5, '5':5}, records
for ps in records.values():
    for name in ps:
        assert list(Path(name).glob('*.wav')), '缺少音频: '+name
        assert list(Path(name).glob('position_source*.txt')), '缺少标签: '+name
out = Path(os.environ['V14_RUN_ROOT'])
(out/'recordings.json').write_text(json.dumps(records, indent=2))
lib = Path(os.environ['LIBRISPEECH_ROOT'])
for split in ('train-clean-100','test-clean'):
    assert (lib/split).is_dir() or (lib/'LibriSpeech'/split).is_dir(), split
print('P0数据目录检查通过；P1前还需确认dev-clean')
PY
~~~

### 4.2 冻结8组检查点，严格加载并建立登记表

此段直接复制执行。若某个checkpoint路径缺失，先找到原文件；不得用同名新权重替换。若strict加载出现残差norm等缺键/多键，记录错误并恢复该检查点对应结构；**禁止strict=False吞掉差异**。表中路径全部为原始运行，不依赖本地paper_rewriting_output目录。

~~~bash
python - <<'PY'
import os, sys, json, hashlib, inspect
from pathlib import Path
import torch
sys.path[:0] = [str(Path('IFAN_Edge/scripts').resolve()), str(Path('IFAN_Edge').resolve()), str(Path('.').resolve())]
from evaluate_stage3_locata import remap_legacy_map_refiner_keys
from ifan_edge.models import IFANModel
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline
out = Path(os.environ['V14_RUN_ROOT'])
root = Path('IFAN_Edge/outputs/stage3')
runs = {
 'c16_dense': 'ifan_stage3_long80_freqblock_paper_original_20260426_155330',
 'saf4of8': 'ifan_stage3_c16_saf_lite_4of8_phase1_20260518_014757',
 'saf3of8': 'ifan_stage3_c16_saf_lite_3of8_phase1_20260518_072249',
 'saf2of8': 'ifan_stage3_c16_saf_lite_2of8_phase1_20260518_125858',
 'c8_dense': 'ifan_stage3_long80_c8_paper_original_20260505_222115',
 'maba_pre': 'ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408',
 'maba_map': 'ifan_stage3_long80_c8_r2_map_maba_20260516_154535',
 'maba_dual': 'ifan_stage3_long80_c8_r2_maba_dual_refine_20260519_191649',
}
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
registry = []
for label, run in runs.items():
    source = (root/run/'checkpoints/best_rmsae.pt').resolve()
    assert source.is_file(), source
    # 仅加载自己的可信检查点，兼容环境中torch.load的默认行为。
    kw = {'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters: kw['weights_only'] = False
    ck = torch.load(source, **kw)
    cfg = IFANTrainingConfig(**ck['training_config'])
    cfg.librispeech_path = str(Path(os.environ['LIBRISPEECH_ROOT']).resolve())
    cfg.baseline_checkpoint_path = str(Path('models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin').resolve())
    pipeline = IFANTrainingPipeline(cfg)
    model = IFANModel(pipeline.model_config)
    state = remap_legacy_map_refiner_keys(ck['model_state_dict'], model_config=pipeline.model_config)
    model.load_state_dict(state, strict=True)
    profile = pipeline.build_model_profile(model)
    if label.startswith('saf'):
        assert any((v == 0).any().item() for k,v in state.items() if k.endswith('weight')), '稀疏checkpoint未见零权重'
    dst = (out/'checkpoints_eval'/(label+'.pt')).resolve()
    assert not dst.exists(), '拒绝覆盖已有评估副本'
    copy = dict(ck)
    copy['model_state_dict'] = state
    copy['training_config'] = dict(ck['training_config'])
    copy['training_config']['librispeech_path'] = cfg.librispeech_path
    copy['training_config']['baseline_checkpoint_path'] = cfg.baseline_checkpoint_path
    copy['_v14_eval_copy'] = {'source':str(source), 'source_sha256':sha(source), 'changes':'pre-readout prefix rename if needed; local dataset/baseline paths only; no tensor removal or retraining'}
    torch.save(copy, dst)
    registry.append({'label':label, 'original_checkpoint':str(source), 'original_sha256':sha(source), 'eval_checkpoint':str(dst), 'eval_sha256':sha(dst), 'selected_epoch':ck.get('epoch'), 'training_config':ck['training_config'], 'model_config':ck.get('model_config'), 'model_profile':profile})
    print(label, 'strict load OK', profile.get('total_params'))
(out/'registry.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
(out/'models.tsv').write_text(''.join(r['label']+'\t'+r['eval_checkpoint']+'\n' for r in registry))
baseline = Path('models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin')
(out/'baseline_sha256.txt').write_text(sha(baseline)+'\n')
PY
~~~

注意：稀疏checkpoint“存在零权重”仅是快速预检，正式还要用对应pruning/mask.pt核查 **11个C16×C16 ConvIco层、每输出通道每8输入连接恰好保留N组**，stem不剪。非零参数、掩码存储开销、稠密参数张量分别报告。

### 4.3 LOCATA：先一条smoke，再全部23段

~~~bash
python IFAN_Edge/scripts/evaluate_stage3_locata.py \
  --checkpoint "$V14_RUN_ROOT/checkpoints_eval/maba_pre.pt" \
  --locata-root "$LOCATA_ROOT" --subset eval --array benchmark2 \
  --tasks 1 --recording recording1 --device cuda \
  --output "$V14_RUN_ROOT/locata/smoke_maba_pre.json" \
  2>&1 | tee "$V14_RUN_ROOT/logs/locata_smoke.log"

while IFS=$'\t' read -r label checkpoint; do
  python IFAN_Edge/scripts/evaluate_stage3_locata.py \
    --checkpoint "$checkpoint" --locata-root "$LOCATA_ROOT" \
    --subset eval --array benchmark2 --tasks 1 3 5 --device cuda \
    --output "$V14_RUN_ROOT/locata/$label.json" \
    2>&1 | tee "$V14_RUN_ROOT/logs/locata_$label.log"
done < "$V14_RUN_ROOT/models.tsv"
~~~

正式命令不要带recording筛选。一次运行一个模型，避免显存竞争；每个报告会同时计算icoCNN，后面必须检查8份报告的baseline逐录音向量一致。checkpoint、subset、array相同还不够，应核对录音集合、有效帧、前端定义和状态。

### 4.4 模拟：同一次调用比较8组，扩大场景数

~~~bash
python - <<'PY'
import os, csv, sys, subprocess
from pathlib import Path
out = Path(os.environ['V14_RUN_ROOT'])
cmd = [sys.executable, 'IFAN_Edge/scripts/evaluate_stage3_simulated.py',
       '--device','cuda','--validation-size','128','--scenario-eval-size','64',
       '--trajectory-seconds','20','--seeds','142','143','144',
       '--output',str(out/'simulated/all_candidates.json')]
for label, ck in csv.reader((out/'models.tsv').open(), delimiter='\t'):
    cmd += ['--checkpoint',ck,'--label',label]
with (out/'logs/simulated_all.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Completed:', out/'simulated/all_candidates.json')
PY
~~~

该脚本按cache_signature复用场景缓存，但它不自动导出逐帧轨迹或全部生成清单。P0先保留当前汇总；正式配对统计和P2机制分析还需补导出。**三个seeds是测试场景生成种子，不是训练重复次数。** 不通过统一`--config`覆盖8个checkpoint，否则可能把宽度、MABA或稀疏配置一起改掉。

### 4.5 P0完成检查及结果打包

~~~bash
python - <<'PY'
import os, json, math, statistics, csv
from pathlib import Path
out = Path(os.environ['V14_RUN_ROOT'])
registry = json.loads((out/'registry.json').read_text())
keys = ['with_silences_rmsae_deg','without_silences_rmsae_deg']
reference = None
summary, recordings = [], []
for item in registry:
    label = item['label']
    d = json.loads((out/'locata'/(label+'.json')).read_text())
    rows = sorted(d['per_recording'], key=lambda x:(x['task'],x['recording'],x['array']))
    assert len(rows)==23 and len({(r['task'],r['recording'],r['array']) for r in rows})==23
    assert {t:sum(r['task']==t for r in rows) for t in [1,3,5]} == {1:13,3:5,5:5}
    b = [(r['task'],r['recording'],r['array'],*[r['baseline'][k] for k in keys]) for r in rows]
    if reference is None: reference=b
    assert [r[:3] for r in b]==[r[:3] for r in reference]
    assert max(abs(x-y) for r,q in zip(b,reference) for x,y in zip(r[3:],q[3:]))<=1e-5, '基线漂移，先查原因再汇总'
    for r in rows:
        assert all(math.isfinite(r[m][k]) for m in ['ifan','baseline'] for k in keys)
        recordings.append({'model':label,'task':r['task'],'recording':r['recording'],'array':r['array'],**r['ifan']})
    for k in keys:
        a=[r['ifan'][k] for r in rows]; base=[r['baseline'][k] for r in rows]
        summary.append({'model':label,'metric':k,'count':23,'recording_mean':statistics.mean(a),'recording_std':statistics.pstdev(a),'baseline_mean':statistics.mean(base),'paired_delta_deg':statistics.mean(x-y for x,y in zip(a,base))})
for name, data in [('locata_recomputed.csv',summary),('locata_recordings.csv',recordings)]:
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print('P0 LOCATA完整且配对基线一致；模拟结果另看all_candidates.json')
PY
~~~

保留整个V14_RUN_ROOT，至少返回registry、环境、日志、全部JSON/MD、逐录音CSV。若较新Torch的加载行为与原环境不兼容，优先恢复已验证版本；确需对自己的可信checkpoint增加weights_only=False时单独记录代码变更。P0跑完先看下面三个问题，再排P1：

- 3/8、2/8是否在Task3/5及困难模拟场景出现稳定退化？
- 4/8、C8在误差与理论/实际成本上是否各有优势？
- 读出前的困难场景收益是否足以解释其与map/dual的真实数据差异？

## 5. P1：需要训练的核心对照

### 5.1 统一训练配置与执行模板

复制配置到本轮输出目录，不修改历史TOML。**P1开始前需准备dev-clean，并补齐第3.2节的数据随机性/互斥清单要求。** 下段只完成可直接配置的部分，不代表已经实现样本清单冻结。

~~~bash
python - <<'PY'
import os, re
from pathlib import Path
out=Path(os.environ['V14_RUN_ROOT'])/'configs'; out.mkdir(exist_ok=True)
src=Path('IFAN_Edge/configs/stage3_long_budget_c8_r2_maba_pre_readout.toml').read_text()
src=re.sub(r'(?m)^test_split\s*=.*$', 'test_split = "dev-clean"',src)
src=re.sub(r'(?m)^librispeech_root\s*=.*$', 'librispeech_root = "'+Path(os.environ['LIBRISPEECH_ROOT']).resolve().as_posix()+'"',src)
src=re.sub(r'(?m)^validation_dataset_size\s*=.*$', 'validation_dataset_size = 128',src)
(out/'train_base.toml').write_text(src)
print(out/'train_base.toml')
PY

train_v14() {
  label="$1"; seed="$2"; shift 2
  python IFAN_Edge/scripts/train_stage3_ifan.py \
    --config "$V14_RUN_ROOT/configs/train_base.toml" \
    --seed "$seed" --device cuda --epochs 80 --phase1-epochs 20 \
    --branch-channels 8 --map-refiner none --weak-map-refiner none \
    --input-ablation-mode none --temporal-module conv \
    --temporal-conv-variant standard_1d \
    --val-size 128 --scenario-eval-size 8 \
    --output-suffix "v14_$label"_s"$seed" \
    --experiment-role "v14_$label" "$@" \
    2>&1 | tee "$V14_RUN_ROOT/logs/train_$label"_s"$seed.log"
}

# 首轮seed42。其余种子用相同命令改为43和44。
train_v14 c8_dense 42
train_v14 c16_dense 42 --branch-channels 16
train_v14 maba_pre 42 --map-refiner maba --map-refiner-position pre_readout
~~~

`train_v14`始终以无MABA为默认，额外参数显式启用目标结构。当前程序在IFAN_Edge/outputs/stage3生成带时间戳运行目录；**从日志记录实际目录和checkpoint，不用glob自动选择“最新”权重**。若改train-size以控制耗时，所有对照同样改变并记录每轮轨迹数、优化器步数与GPU小时；不把缩短训练标为原完整预算。

新训练全部结束后，用checkpoint内配置建立评估副本：只把`training_config.test_split`从dev-clean改为test-clean，保留原字段与override记录；严格加载并按第4节运行最终评估。不要重用P0脚本中的历史路径表来评价新模型。

具体复用方式：为新实验另建一个结果目录，重复4.1的输出目录设置；把4.2的runs字典替换为本轮实际运行目录（label包含训练seed），保留严格加载和SHA记录，在torch.save(copy, dst)之前加入下段。然后原样运行4.3–4.5。不得修改原best.pt的训练配置来伪装历史数据使用方式。

~~~python
# 仅用于P1/P2新训练的最终测试副本，不加到P0历史复评。
assert ck['training_config']['test_split'] == 'dev-clean'
copy['training_config']['test_split'] = 'test-clean'
copy['_v14_eval_copy']['heldout_override'] = {
    'training_validation_split': 'dev-clean',
    'final_simulation_split': 'test-clean',
    'selection_rule': 'checkpoint selected on dev-clean before final evaluation',
}
~~~

当前模拟脚本会连带计算名为validation的测试缓存指标：在这个最终测试副本中它来自test-clean，只作测试汇总，**不得再据此换best.pt**。真正的选模验证指标读取原训练目录history/summary。

### 5.2 P1-A：剪枝与通道收缩的公平对照

目标：检验“同一C16母模型、同样20轮适配下，剪多少、损失多少；与规则C8成本相比是否值得”。

每个训练种子先有C16和C8的80轮母模型。严格的80+20预算使用各自**last.pt（epoch=80）**作为适配父模型；所有C16稀疏率与C16稠密继续训练共用同一个父模型。

| 组 | 父模型 | 新阶段 | 差异 |
|---|---|---|---|
| dense16_ft20 | C16同seed的last.pt | 20轮 | 不施加掩码 |
| saf4_ft20 | 同一C16 last.pt | 20轮 | 4/8 |
| saf3_ft20 | 同一C16 last.pt | 20轮 | 3/8 |
| saf2_ft20 | 同一C16 last.pt | 20轮 | 2/8 |
| dense8_ft20 | C8同seed的last.pt | 20轮 | 规则稠密C8 |

全部用`--ifan-init-checkpoint`开始新适配阶段，重新初始化Adam，学习率1e-5、phase1_epochs=0、batch10/micro1、SNR5–30 dB；不混用resume和init。若先用历史best做快速试验，标签标为historical_parent_ft20并记录真实父epoch（历史C16最佳68、C8最佳64），**不能声称实际经历了固定80+20轮**。不能把同一个seed42父模型只换微调seed三次，称为三次独立预训练。

~~~bash
# 以下路径必须替换为本轮对应seed的实际run；先核对epoch=80。
export C16_PARENT="IFAN_Edge/outputs/stage3/实际C16运行/checkpoints/last.pt"
export C8_PARENT="IFAN_Edge/outputs/stage3/实际C8运行/checkpoints/last.pt"

train_v14 dense16_ft20 42 --branch-channels 16 \
  --epochs 20 --phase1-epochs 0 --lr-phase1 1e-5 --lr-phase2 1e-5 \
  --ifan-init-checkpoint "$C16_PARENT"
for keep in 4 3 2; do
  train_v14 saf"$keep"_ft20 42 --branch-channels 16 \
    --epochs 20 --phase1-epochs 0 --lr-phase1 1e-5 --lr-phase2 1e-5 \
    --ifan-init-checkpoint "$C16_PARENT" --saf-lite --saf-lite-keep-per-8 "$keep"
done
train_v14 dense8_ft20 42 --branch-channels 8 \
  --epochs 20 --phase1-epochs 0 --lr-phase1 1e-5 --lr-phase2 1e-5 \
  --ifan-init-checkpoint "$C8_PARENT"
~~~

不能将C16父权重直接载入C8：它们形状不同，C8需自己的母模型。固定验证集从20轮中选best，不看LOCATA挑轮次。可追加每个epoch记录验证曲线，确认强剪枝是否仍未收敛；延长预算必须对稠密控制组同步延长。

**输出和结论：**每组四场景、困难均值、LOCATA两口径/任务/录音、参数、MAC、有效MAC、激活内存、训练成本。理论有效全网MAC为“未剪层MAC + 保留连接的ConvIco MAC”，不是只报剪枝部分。旧2/8、3/8、4/8全网理论代理约122.780/178.906/235.031 M，而其稠密执行仍约459.533 M；以新profile重算为准。只有误差与**实测**速度/内存共同占优，才可说路线更适于部署。若只证明稀疏率增加的精度代价，就限定到当前剪枝准则、模型和训练条件。

### 5.3 P1-B：MABA位置和组件

所有组固定C8/r2，采用相同训练配方和种子。无MABA的c8_dense复用第5.1节同seed运行，不另重复训练。

~~~bash
train_v14 maba_map 42 --map-refiner maba --map-refiner-position pre_softargmax
# maba_pre已在5.1节执行，不要重复。
train_v14 maba_dual 42 --map-refiner maba --map-refiner-position pre_readout \
  --weak-map-refiner maba --weak-map-maba-d-model 8 --weak-map-maba-state-dim 4 \
  --weak-map-maba-conv-kernel 3 --weak-map-maba-dropout 0.0 --weak-map-maba-no-state
train_v14 maba_pre_no_state 42 --map-refiner maba --map-refiner-position pre_readout \
  --map-maba-no-state
train_v14 maba_pre_no_gate 42 --map-refiner maba --map-refiner-position pre_readout \
  --map-maba-no-gate
~~~

主模块固定d=16、state=8、kernel=3、dropout0.1、残差开启；双位置的弱模块配置见命令。这是**同训练配方的结构对照，非等参数/等MAC对照**，各组重新profile。若要证明优势不是简单增加参数，再补参数预算接近的因果Conv1d残差块，当前CLI未提供，不用“无状态”冒充参数匹配基线。

必须同时报告：总体、Task1/3/5、四模拟场景、参数/MAC增量、CPU/GPU模型执行时间。读出前若只在困难模拟或Task5有优势，结论限定于这些条件。no-state/no-gate要重训练；对已训练模型临时关状态只能另标为推理扰动，不能替代消融。

### 5.4 P1-C：双特征信息消融

~~~bash
train_v14 c8_phat_only 42 --input-ablation-mode phat_only
train_v14 c8_lms_only 42 --input-ablation-mode lms_only
# 双特征控制组是5.1节c8_dense，省去重复训练。
~~~

固定无MABA、相同C8主干和训练预算。回答“两种特征是否在低SNR/高混响下互补”；不能把置零导致的效果写成硬件删枝收益。若希望同时讨论单分支节省资源，另实现真实单分支模型并分表比较。

### 5.5 训练数量与调度建议

- 先完成P0；再跑seed42的P1-A/B/C，检查脚本、数据和趋势。
- 正式关键集合每个seed：3个80轮基础模型（C16/C8/pre），5个20轮适配（含3种剪枝）、4个80轮MABA对照（map/dual/no-state/no-gate）、2个80轮输入对照，共14个运行；三个seed共42个运行。已有完全相同代码、数据和配方的合格运行可复用。
- GPU紧张时先完成压缩与MABA的关键组，其余先保留单seed探索；论文明确未完成重复训练的范围。不承诺单次运行小时数，先实测epoch时间再排队。
- 同卡避免并发训练与计时实验；多个GPU按独立实验分配，各组保留相同硬件和软件版本记录。

## 6. P2：按论文主张追加，不阻塞P0

### 6.1 宽度和分辨率

在相同r2、无MABA下追加C4、C12；与已跑的C8/C16构成4点曲线：

~~~bash
train_v14 c4_dense 42 --branch-channels 4
train_v14 c12_dense 42 --branch-channels 12
~~~

r3先复制train_base.toml，仅改[model]的r=3，独立调用train_stage3_ifan.py并保持其余训练参数相同。不要直接拿旧r3的另一套baseline向量作比较。扫描报告RMSAE—MAC、RMSAE—激活内存和RMSAE—推理时间；C8只有在给定资源/误差约束下合理，不能因选了它就称最优。

**停止规则：**验证集用于选折中点；LOCATA eval不用于决定下一轮更有利参数。若不预设明确资源上限，只能描述非支配候选和折中，不能宣称求得最优网络。

### 6.2 残差与注意力（需补实现）

当前CLI没有这些开关，不存在可直接复制的`--no-attention`或`--no-residual`主干参数。先以C8无MABA为控制，新增独立实验配置，不改默认模型：

| 变体 | 明确定义 | 检验问题 |
|---|---|---|
| 无残差增强 | 每分支enhanced=D（stem直通特征），其余融合公式保持，记录增益尺度变化 | 分支内增强是否有价值 |
| 固定融合 | 每分支输出D+E，取消A(E)网络，仅相加两分支 | 自适应权重是否有额外作用 |
| 共享注意力 | 两分支各自计算A(E)，共享同一组权重 | 当前方法 |
| 独立注意力 | 两分支相同注意力结构，参数互不共享 | 参数共享带来何种精度/成本折中 |

新增结构需：forward shape/finite测试、梯度可达、计数变化与开关一致、checkpoint保存配置并能重载；再做同预算重训练。无残差变体的融合幅值有变化时，解释该干预包含什么，不能把收益全部归于单一机制。若无法补这些实验，论文不把残差/共享注意力单独列为已验证创新点。

### 6.3 逐帧机制分析（需补导出）

每段录音保存timestamps、预测/真值单位方向、角误差、VAD有效掩码、状态复位标记；不要仅保存三个任务均值。对相同录音展示C8、pre、map/dual的轨迹和误差，场景选择规则应预先固定（例如各任务recording1并附全录音统计），避免只选最好例子。

转向片段以真值角速度分层：阈值从开发数据确定并冻结；报告高角速度帧RMSAE、误差p95和峰值。若定义恢复时间，先固定阈值和连续帧规则；不能看到曲线后再选有利阈值。静音恢复按真实VAD从0到1的转折分析，保持启动帧规则一致。

### 6.4 软件成本与流式状态

现有benchmark可直接做同录音整段软件成本对照，例如：

~~~bash
python IFAN_Edge/scripts/benchmark_stage3_inference.py \
  --checkpoint "$V14_RUN_ROOT/checkpoints_eval/maba_pre.pt" \
  --target ifan --mode model_only --device cuda \
  --locata-root "$LOCATA_ROOT" --subset eval --array benchmark2 \
  --tasks 1 3 5 --max-recordings 23 --warmup 30 --repeats 100 \
  --output "$V14_RUN_ROOT/maba_pre_gpu_model_only.json"
~~~

同样比较C16、C8、4/8/3/8/2/8、pre；CPU/GPU分开；model_only与end_to_end分开。该脚本整段录音计时，记录实际T及总帧数，**均摊时间不是在线单帧时延**。现有脚本给p90而非p95，p95/显存峰值需额外采样实现，不能改列名冒充。

零权重模型仍运行稠密PyTorch算子，速度没有下降是有效结果，不是实验出错。若报告稀疏加速，必须有实际跳零kernel，当前实验不包含。

流式验证分三组：整段forward参考；每6帧独立reset的负对照；正确保留所有因果卷积历史及MABA状态的chunk实现。第三组当前未提供状态接口，需实现。每段新录音只reset一次；保持同一LMS前端状态。先验证无NaN及坐标/方向一致，再跑LOCATA，不用仅保存MABA状态替代整网历史缓存。此项是实时部署主张的前置条件。

## 7. 返回结果清单与论文写法

### 7.1 必须返回

1. 环境：git commit、dirty patch、pip freeze、GPU/驱动、实际命令和运行日志。
2. 模型：配置快照、父checkpoint SHA/实际epoch、best/last、训练seed、训练预算/样本数/更新步数/GPU小时、参数及MAC口径。
3. 测试：23段完整LOCATA JSON/MD、逐录音CSV、模拟测试种子/每场景数/各场景结果；新实验单独记录dev验证与test最终测试清单。
4. 稀疏：mask.pt、pruning_summary.json、每层保留组数、未剪层和剪层MAC；实际计时如做过则另列。
5. 机制：逐帧预测/标签/掩码（实施P2-C时）、轨迹图及选择规则；不只发截图或一个平均数。
6. 失败：加载不兼容、非有限值、缺录音、OOM、未收敛都保留日志，结果填null/失败原因，不填0。

建议结果目录保持为IFAN_Edge/outputs/algorithm_supplement_时间戳。大权重和音频不必放Git，用现有共享路径/归档返回；文档和小型汇总可以提交。服务器新代码若为补实验实现，单独提交并在结果中记录commit，避免与本次文档提交混淆。

### 7.2 每组实验怎样形成结论

| 问题 | 数据充分后可写 | 暂时不可写 |
|---|---|---|
| 为何不用N/8 | 在指定任务/预算下，高稀疏率的精度与真实实现成本不满足约束 | 结构化剪枝普遍无效 |
| 为何用C8 | 在给定资源与误差约束下提供合理折中，结构规则且激活减少 | C8是全局最优；MAC下降等于实测加速 |
| 为何用读出前MABA | 相对C8、map、dual，在特定困难条件/成本维度有可重复收益 | 所有任务领先；整体差异都来自状态机制 |
| 双特征/注意力 | 同配方训练消融支持某组件的增量 | 两张响应图就证明互补性；参数多就是原因 |
| 能否实时 | 维护状态的流式实现、实际输入处理和计时共同满足约束 | 六帧平均时间足够小就已实现在线系统 |

**最先交回P0即可继续修改论文结论和排P1；不需要等所有可选实验跑完。** 最终结论允许实验改变原先路线，不能以“必须证明我们的方案最好”为筛选规则。
