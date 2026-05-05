# Vitis HLS 终端流程（VSCode）

## 1. 在 VSCode 终端进入目录

```powershell
cd g:\3DSLED\icocnn\hls_src
```

## 2. 配置 Vitis 环境（若 `vitis_hls` 不在 PATH）

```powershell
call <Vitis安装目录>\settings64.bat
```

## 3. 一键运行

默认（`quick`）：`csim + csynth`

```powershell
.\run_hls.bat
```

## 4. 常用运行模式

```powershell
.\run_hls.bat csim
.\run_hls.bat synth
.\run_hls.bat cosim
.\run_hls.bat export
.\run_hls.bat all
```

## 5. 自定义参数

参数顺序：

```text
run_hls.bat <mode> <part> <clock_ns> <project> <solution> <top>
```

示例：

```powershell
.\run_hls.bat quick xc7k325tffg900-2 5.0 layer0_hls_prj sol1 conv_ico_layer0
```

## 6. 报告位置

- 原始综合报告：
  - `hls_src/<project>/<solution>/syn/report/*_csynth.rpt`
  - `hls_src/<project>/<solution>/syn/report/*_csynth.xml`
- 汇总报告（自动生成）：
  - `hls_src/hls_reports/latest_summary.md`
  - `hls_src/hls_reports/<project>_<solution>_<timestamp>/summary.md`

