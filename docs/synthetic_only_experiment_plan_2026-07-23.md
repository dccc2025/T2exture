# T2exture 合成数据实验计划

## 0. 当前边界

- 当前阶段只做 synthetic 数据集实验，不做 real transfer / real finetune。
- 论文主表、消融表、部署表统一使用 scene-disjoint split；正式运行时 split 文件位于 `dataset_roi/train.txt`, `dataset_roi/valid.txt`, `dataset_roi/test.txt`。
- 当前 split 已落地：train / valid / test = 20 / 4 / 8 scenes。
- `binoculars` 因 ROI 统计中横向运动轨迹异常接近全宽而从正式 split 中剔除；`bunny` 已从 valid 挪到 train，保持 train 为 20 scenes。
- 后续正式实验切换到 `dataset_roi`：原始 `dataset` 只作为 source root，`dataset_roi` 作为唯一正式 `--data-root`。公开和复现时将 `dataset_roi` 视为正式 synthetic 数据集的固定 `640x960` frame protocol，不在实验表中混用原始 source root。
- `third_party/` 只作为本地下载和运行中转，不进入 git；重要权重统一放在 `pretrained/<method>/` 子目录，AMT 当前训练结束后再从根目录迁移到 `pretrained/AMT/`。
- 已确认原 `pretrained/IFRNet.pth` 实际是 LiteFlowNet 权重，已重命名为 `pretrained/LiteFlowNet.pth`；不能用于表 1 的 IFRNet baseline。
- 正式实验不复用旧 split 的数值，只复用旧实验给出的方向性判断。
- 已确认 baseline 最终名单：IFRNet、SGM-VFI、BiM-VFI、GIMM-VFI-F、AMT-L、Ours-L。
- InterpAny、LDF-VFI、EDEN 彻底退出当前正式实验，不再进入主表或消融表。
- 已确认主 loss 保留 `0.001 * L_flow`，当前实验先按固定协议跑，暂时不做 `w/o L_flow`。
- 已确认 Table 4 需要生成全套 active-sparsity pseudo-flow set：`flow/s02` 到 `flow/s11`；这些 set 应在 `dataset_roi` 下生成或裁剪，不写回原始 `dataset`。
- 已确认改为离线 ROI 裁剪数据集；`dataset_roi` 生成并确认后，训练阶段仍采用 VFI-style random crop，验证/测试使用完整 ROI。

## 0.1 工作站和执行原则

- 当前工作站电源计划已切换到“卓越性能”，可以按长时间满负载实验使用。
- 默认允许全功率运行，不因为耗电、风扇或轻微资源占用而主动降速。
- 可以并行执行互不冲突的任务，例如 ROI preview、baseline wrapper 准备、指标脚本检查、文档更新和轻量统计。
- GPU 长训任务谨慎并行：同一张 GPU 上不同时跑多个大模型训练；训练时可以并行 CPU/I/O 侧整理和轻量检查。
- 优先节省总实验时间，但不能牺牲实验公平性、可复现性和可视化审计。
- 同一张表必须统一 split、ROI、输入分辨率、训练 crop、loss、评估指标和 checkpoint 选择规则。
- 所有正式实验必须保留 checkpoint、预测帧、误差图、aggregate metrics、scene/frame CSV、manifest 和可视化视频。
- 可以自动推进无争议任务，减少人工确认；涉及删除/覆盖原始数据或权重、外部上传、实验协议变化、OOM 后降级、baseline 代码源不确定、指标异常时必须先停下来确认。

## 1. 固定训练和评估协议

### 数据

source root：

```text
dataset/
  train.txt
  valid.txt
  test.txt
  sim/<scene>/texture/*.npy
  sim/<scene>/passive/*.npy
  flow/s10/<scene>/<left>_<target>_<right>.npz
```

formal experiment root：

```text
dataset_roi/
  train.txt
  valid.txt
  test.txt
  roi_manifest.json
  roi_preview/<scene>.png
  sim/<scene>/texture/*.npy
  sim/<scene>/passive/*.npy
  flow/s10/<scene>/<left>_<target>_<right>.npz
```

- 原始 `dataset` 不再直接进入正式训练，只用于生成 `dataset_roi`。
- 正式训练、验证、测试和 baseline comparison 统一使用 `--data-root dataset_roi`。
- `dataset_roi` 中原始帧仍放在 `sim/<scene>/texture` 和 `sim/<scene>/passive`。
- `dataset_roi` 中 pseudo-flow 仍按 AMT 风格单独放在 `flow/<flow_set>`，不塞进 `sim/<scene>`。
- 默认协议使用 `flow/s10`，表示 active anchors 间隔 10 帧。
- 默认样本为 `left=1,11,...,161`，`right=left+10`，中间目标 `target=left+1...left+9`。
- source 分辨率为 `768 x 1024`。
- `dataset_roi` 只做 crop / pad，不做 resize；这样已有 flow 只需要同步裁剪空间区域，不需要缩放 flow 数值。
- 每个 scene 使用一个固定 crop box，覆盖目标物体完整运动轨迹；禁止逐帧动态裁剪。
- 所有 scene 输出统一空间分辨率，目标尺寸先定为 `640 H x 960 W`，仍需通过 ROI preview / QC 确认。

### Loss

正式主方法先固定为：

```text
L = 1.0 * L_charbonnier + 0.1 * L_css + 0.001 * L_flow
```

- `L_charbonnier`：主重建项，使预测 texture 接近 GT texture。
- `L_css`：census-style 局部结构约束。
- `L_flow`：LiteFlowNet pseudo-flow distillation，只作为 AMT 中间 flow 的辅助监督。
- `L_edge` 和 `L_temp` 不作为当前主方法 loss；旧结果显示二者没有稳定提升。
- `w/o L_flow` 暂时不做；当前优先完整跑固定主协议。
- 论文叙事中不要把 `L_flow` 写成核心贡献；核心贡献放在 passive context、adapter 跨域适配和 synthetic finetuning protocol。

### 训练超参数

`dataset_roi` 确认生成后，正式 `train.yaml` 应执行。训练 crop 按当前 T2exture 数据实际尺寸和目标运动轨迹选择 `384`，不照搬 AMT 在 Vimeo90K 上的 `224`：

```yaml
passive_context: 4
active_stride: 10
sample_passive_context:
require_dataset_roi: true
pseudo_flow_dir: flow/s10
adapter_iterations: 10000
finetune_iterations: 5000
batch_size: 4
valid_batch_size: 1
eval_batch_size: 1
crop_size: 384
adapter_lr: 0.0002
finetune_lr: 0.00005
layerwise_lr_decay: 0.8
num_workers: 16
pin_memory: true
persistent_workers: true
prefetch_factor: 4
seed: 2026
valid_interval: 500
loss:
  charbonnier: 1.0
  css: 0.1
  flow: 0.001
```

当前工作站为 Threadripper PRO 9985WX 64C/128T、256GB RAM、RTX 4090 48GB。训练阶段使用 `384` crop，因此默认训练 `batch_size` 保持 4；验证/测试阶段使用完整 `640x960` ROI，AMT all-pairs correlation 显存压力更高，因此 `valid_batch_size` 和 `eval_batch_size` 先保持 1，避免 full-frame OOM。短 benchmark 显示 AMT-L finetune 在 `batch_size=4` 时约 `0.13s/step`、峰值约 `5.2GB`；提高到 8/16/24 会让单步时间近似线性增加，而正式训练按 iterations 计数，因此不作为默认加速方案。`num_workers=16`、`pin_memory=true`、`persistent_workers=true`、`prefetch_factor=4` 是当前默认吞吐配置；64 workers 启动和调度开销更大，不建议使用。

两阶段含义：

```text
0 - 10000 iterations:
  freeze AMT backbone
  train adapter + passive geometry/fusion modules

10000 - 15000 iterations:
  keep adapter/passive modules trainable
  unfreeze selected rear AMT modules
  use layer-wise learning-rate decay
```

AMT-S / AMT-L / AMT-G 都使用同一训练协议。AMT-G 的 `update2` 在官方结构里拆成 `update2_low` 和 `update2_high`，因此第二阶段 refine scope 需要单独映射，不能直接照搬 AMT-L 的模块名。

### 指标

所有主表统一输出：

```text
PSNR, SSIM, Edge-FI@2px, IE, NIE
```

部署表额外输出：

```text
Latency, FLOPs, Params, Trainable Params
```

旧实验里的 `MAE / LPIPS` 不进入当前主表；当前是灰度 texture 恢复任务，LPIPS 不是首选指标。

### Artifact Contract

每个正式实验必须保留 checkpoint、指标、逐帧输出和可视化：

```text
outputs/final/<table_id>/<exp_id>/
  best.pt
  last.pt
  config.json
  pred/<scene>/*.png
  err/<scene>/*.png
  metrics.json
  scene.csv
  frame.csv
  manifest.json
  vis/png/*.png
  vis/video/*.mp4
```

推荐实验目录 ID：

原则：目录名可以稍长，但必须一眼能看懂；不要使用 `ctx`, `t2x`, `gimmf`, `van` 这类后续容易遗忘的缩写。统一使用小写 kebab-case。

| 类型 | 示例 |
|---|---|
| prior comparison | `table01_prior/ifrnet`, `table01_prior/gimm-vfi-f`, `table01_prior/amt-l-vanilla`, `table01_prior/ours-l` |
| AMT size ablation | `table02_amt_size/amt-s-vanilla`, `table02_amt_size/amt-s-t2texture`, `table02_amt_size/amt-l-t2texture`, `table02_amt_size/amt-g-t2texture` |
| passive context | `table03_passive_context/passive-refs-00`, `table03_passive_context/passive-refs-04`, `table03_passive_context/passive-refs-10` |
| active sparsity | `table04_active_sparsity/active-stride-02`, `table04_active_sparsity/active-stride-10`, `table04_active_sparsity/active-stride-11` |
| deployment | `table05_deployment/amt-l-t2texture` |

`kf.pdf` 不作为默认输出。论文图后续从 `vis/png/` 选关键帧单独导出。

## 2. 已实现并验证的能力

### 数据和 pseudo-flow

- `dataset/train.txt`, `dataset/valid.txt`, `dataset/test.txt` 已是 20 / 4 / 8 scenes。
- 旧 `dataset/sim/<scene>/flow` 的 5045 个 `.npz` 已 hardlink 迁移到 `dataset/flow/s10/<scene>`。
- 当前 `dataset/flow/s10` 文件数：5045。
- source `dataset/flow/s10` 对当前 split 的覆盖检查通过：
  - train: 3040 samples
  - valid: 608 samples
  - test: 1216 samples
- 每个 flow 文件保存 `flow0` 和 `flow1`，shape 为 `[2, 768, 1024]`，dtype 为 `float32`。
- `flow_generation/pseudo_flow_index.py` 已支持 AMT-style 路径，并保留旧 `sim/<scene>/flow` fallback。
- 一次性迁移脚本已删除。
- 主协议 `s10` 在 ROI 数据集中通过裁剪复用已有 `dataset/flow/s10`；Table 4 的其他 stride 后续再用 `flow_generation/generate_liteflownet_flow.py` 生成。

### 实验数据快照

| 项目 | 当前值 |
|---|---|
| source root | `dataset` |
| formal root | `dataset_roi` |
| source resolution | `768 H x 1024 W` |
| formal ROI resolution | `640 H x 960 W` |
| split | train / valid / test = 20 / 4 / 8 scenes |
| total formal scenes | 32 |
| frames per scene | 180 texture + 180 passive |
| default active stride | 10 |
| default passive refs | 4 total refs |
| train / valid / test samples | 3040 / 608 / 1216 |
| source `flow/s10` files | 5045 |
| ROI `flow/s10` files | 4892 |
| ROI previews | 32 PNGs |
| max texture union bbox | `599 H x 753 W` |
| available Table 4 flow sets | `flow/s10` |
| missing Table 4 flow sets | `flow/s02...s09`, `flow/s11` |

正式 split scene：

```text
train: adjustable_wrench, alarm_clock_01, american_football, armadillo, bananas, beast, bunny, boulder_01, brass_blowtorch, carrot_cake, carved_wooden_elephant, CashRegister_01, cassette_player, ceramic_pot, cheburashka, cow, Drill_01, industrial_microscope, max-planck, ogre
valid: korean_fire_extinguisher_01, rocker-arm, vintage_electric_kettle, xyzrgb_dragon
test: spot, teapot, hand_truck, beetle, boombox, Camera_01, metal_toolbox, vintage_video_camera
```

### 训练和模型

- `model/__init__.py` 已支持 `amt-s / amt-l / amt-g` selector。
- 已有独立 wrapper：
  - `model/t2texture_amt_s.py`
  - `model/t2texture_amt.py`
  - `model/t2texture_amt_g.py`
- `train.py` 已支持：
  - iteration-based training
  - `adapter` / `finetune` 两阶段
  - `--resume`
  - fresh `infinite_loader()`，避免 `itertools.cycle(DataLoader)` 缓存旧 batch
  - layer-wise learning-rate decay
  - `torch.set_float32_matmul_precision('high')`
  - `torch.backends.cudnn.benchmark = True`
- `train.yaml` 已写入默认 `pseudo_flow_dir: flow/s10`。
- 当前 `train.yaml` 已写入正式训练 crop：`384`，并通过 `require_dataset_roi: true` 防止正式训练误用原始 `dataset`；验证/测试不 crop，使用完整 `640x960` ROI。

### ROI 数据集状态

- `dataset_roi` 已生成，固定输出尺寸为 `640 H x 960 W`。
- `dataset_roi/train.txt`, `dataset_roi/valid.txt`, `dataset_roi/test.txt` 已规范化为无缩进、无 BOM 的 scene 列表，split 仍为 20 / 4 / 8。
- `dataset_roi/roi_manifest.json` 和 `dataset_roi/roi_preview/<scene>.png` 已生成，可用于后续人工抽查 ROI。
- `dataset_roi/flow/s10` 已由已有 `dataset/flow/s10` 同步裁剪得到，不需要重新跑 LiteFlowNet。
- 当前覆盖检查通过：train 3040 samples，valid 608 samples，test 1216 samples；texture / passive / flow shape 均为 `640 x 960`。
- 不覆盖原始 `dataset/flow/s10` 全分辨率缓存；它保留为 source，正式训练读取裁剪后的 `dataset_roi/flow/s10`。
- 当前 32 个正式 scene 的最大 texture union bbox 高度为 599 px，最大宽度为 753 px，均能被 `640 H x 960 W` 固定 ROI 覆盖。
- ROI 仍需通过 preview 抽查，目标是确认目标物体完整运动轨迹保留在画面中，且所有 scene 空间分辨率一致。
- 当前目标输出尺寸：`640 H x 960 W`。
- `binoculars` 已从正式 split 中剔除，不再用它决定 ROI 宽度。

### 评估和可视化

- `eval.py` 已支持 checkpoint 评估并导出：
  - `pred/`
  - `err/`
  - `metrics.json`
  - `scene.csv`
  - `frame.csv`
  - `manifest.json`
- `vis.py` 已支持从 eval 结果导出：
  - `vis/png/`
  - `vis/video/`
- `metrics/` 已有：
  - `PSNR`
  - `SSIM`
  - `Edge-FI@2px`
  - `IE`
  - `NIE`
  - parameter count helpers

### 执行前 preflight

- 英文执行 runbook 位于 `docs/formal_experiment_runbook.yaml`，用于正式长跑前检查路径、默认协议、split、ROI 尺寸和 artifact contract。
- `scripts/preflight_formal.py` 已新增为一键 preflight 脚本，不启动训练，只检查 required paths、`train.yaml` 与 runbook 是否一致、`dataset_roi` split / sample count / tensor shape / `flow/s10` 覆盖。
- 正式长跑前先执行 runbook 中的 `preflight_command`，再执行训练、评估和可视化命令。

### 指标同步和输出路径

- 所有正式结果必须写入 `outputs/final/<table_id>/<exp_id>/`。
- 每次训练 / 评估 / 可视化完成后，重新运行 `scripts/sync_formal_metrics.py`，同步生成：
  - `outputs/final/_summary/metrics_summary.csv`
  - `outputs/final/_summary/metrics_summary.md`
  - `outputs/final/_summary/by_table/table01_prior.csv`
  - `outputs/final/_summary/by_table/table01_prior.md`
  - `outputs/final/_summary/by_table/table02_amt_size.csv`
  - `outputs/final/_summary/by_table/table02_amt_size.md`
  - `outputs/final/_summary/by_table/table03_passive_context.csv`
  - `outputs/final/_summary/by_table/table03_passive_context.md`
  - `outputs/final/_summary/by_table/table04_active_sparsity.csv`
  - `outputs/final/_summary/by_table/table04_active_sparsity.md`
  - `outputs/final/_summary/by_table/table05_deployment.csv`
  - `outputs/final/_summary/by_table/table05_deployment.md`
- 当前汇总脚本已覆盖表 1 到表 5 的 34 个正式行；当前状态为 `done=4, missing=30`。
- 已补官方 AMT-S/L/G vanilla 评估入口：`scripts/eval_amt_vanilla.py`，输出 `pred/`, `err/`, `metrics.json`, `scene.csv`, `frame.csv`, `manifest.json`，和 T2exture `eval.py` 的 artifact contract 对齐。
- 已完成 AMT vanilla 正式评估和 sample 可视化：
  - 表 1 AMT-L vanilla：`outputs/final/table01_prior/amt-l-vanilla/test`
  - 表 2 AMT-S vanilla：`outputs/final/table02_amt_size/amt-s-vanilla/test`
  - 表 2 AMT-L vanilla：`outputs/final/table02_amt_size/amt-l-vanilla/test`
  - 表 2 AMT-G vanilla：`outputs/final/table02_amt_size/amt-g-vanilla/test`

## 3. 已解决的旧问题

- pseudo-flow 不再默认写入 `dataset/sim/<scene>/flow`；source 缓存路径为 `dataset/flow/<flow_set>/<scene>`，正式 ROI 缓存路径为 `dataset_roi/flow/<flow_set>/<scene>`。
- 已有 5045 个旧 flow 不是坏数据，格式和覆盖都正常；问题只是旧路径组织，现在已迁移到 `dataset/flow/s10`，后续作为 `dataset_roi/flow/s10` 的裁剪源。
- Table 3 和 Table 4 已明确区分：
  - Table 3 测 passive reference 数量。
  - Table 4 测 active frame sparsity。
- Table 4 不再使用 active/passive proportion 的表述，改成 `Passive frames between active anchors`。
- AMT-G 的 refine scope 已明确要适配 `update2_low / update2_high`。
- 真实数据迁移已从当前计划移除，后续单独规划。
- 旧 Ours-v0、edge/temp/flow 探索结果只作为方向性参考，不直接填最终表。

## 4. 仍未完成的工作

这些是正式长跑前或正式实验阶段需要继续补齐的工作：

1. Baseline wrapper / runner
   - IFRNet wrapper 已补完整评估入口 `scripts/eval_ifrnet.py`；官方 VFI checkpoint 已放在 `pretrained/IFRNet/IFRNet_Vimeo90K.pth`，GoPro 权重只保留不用作表 1 默认。
   - GIMM-VFI-F 官方代码已下载到 `third_party/GIMM-VFI`，代码源为 `https://github.com/GSeanCDAT/GIMM-VFI`。
   - GIMM-VFI-F wrapper 需要补完整评估入口，使用 `pretrained/GIMM-VFI-F/flowformer_sintel.pth`, `pretrained/GIMM-VFI-F/gimm.pt`, `pretrained/GIMM-VFI-F/gimmvfi_f_arb.pt`。
   - AMT-L vanilla 已统一到同一套 test split、指标和 artifact contract。
   - SGM-VFI / BiM-VFI 需要统一到同一套 test split、指标和 artifact contract。
   - InterpAny / LDF-VFI / EDEN 不再补 wrapper，不跑正式表。

2. Vanilla AMT-S/L/G 评估入口
   - `scripts/eval_amt_vanilla.py` 已补，可跑 AMT-S/L/G vanilla。
   - 表 1 的 AMT-L vanilla 和表 2 的 AMT-S/L/G vanilla 可直接从该入口评估。
   - vanilla 不使用 passive context，也不训练 adapter；评估 sample window 与默认协议一致。
   - 参数量统计口径需要和 T2exture wrapper 区分：`official backbone params` vs `T2exture total params`。

3. Passive context ablation
   - 代码已支持 `passive_context=0 / 2 / 4 / 6 / 8 / 10`。
   - 表 3 正式配置建议所有行统一设置 `sample_passive_context: 10`，保证 no-passive 和多 passive 行使用同一 target-frame 评估窗口。
   - 仍需为每一行生成独立 config / output 目录，并重训重评。

4. Active frame sparsity ablation
   - `TextureDataset` / `train.py` / `eval.py` 已支持配置化 `active_stride`，并使用 `time = offset / active_stride`。
   - 每个 stride 如果保留 `L_flow`，都需要对应 `dataset_roi/flow/sXX` pseudo-flow set。
   - `generate_liteflownet_flow.py` 已支持 `--active-stride`；未显式传 `--flow-set` 时会自动写入对应 `flow/sXX`。

5. Ablation switches
   - 当前不实现 `w/o L_flow`；主 loss 先固定为 `L_char + 0.1 L_css + 0.001 L_flow`。
   - passive ablation 需要配置化关闭 passive input。
   - refine scope ablation 需要配置化选择 decoder/update/comb 组合。
   - 消融入口建议和正式 Ours-L 主线分开，避免污染主训练协议。

6. Deployment profiling
   - 目前只有参数量统计 helper。
   - 还缺统一 latency / FLOPs profiling 入口。
   - latency 需要固定输入分辨率、batch size、warmup 次数、重复次数和 GPU 型号。

7. Experiment runner
   - 英文 runbook 已列出关键手动命令和所有正式输出目录。
   - 还缺一键顺序运行表 1 / 表 2 / 表 5 的 runner。
   - runner 需要保存完整命令、git 状态、配置快照和 checkpoint 路径。
   - 当前可以先按 runbook 手动逐条跑，但正式结果必须按 artifact contract 收敛到 `outputs/final/...`。
   - `scripts/sync_formal_metrics.py` 已补，用于每次实验结束后同步统计指标。
8. Table 4 pseudo-flow sets
   - `dataset_roi/flow/s10` 已完成。
   - `dataset_roi/flow/s02...s09,s11` 仍未生成；生成后需要再次跑 preflight 或专门覆盖检查。

## 5. 正式实验队列

### 表 1：Prior Work Comparison，放 5.2.2

| Method | Params | PSNR | SSIM | Edge-FI@2px | IE | NIE |
|---|---:|---:|---:|---:|---:|---:|
| IFRNet [CVPR'22] | 5.0M |  |  |  |  |  |
| SGM-VFI [CVPR'24] | 20.8M |  |  |  |  |  |
| BiM-VFI [CVPR'25] | 6.88M |  |  |  |  |  |
| GIMM-VFI-F [NeurIPS'24] | 待统一脚本统计 |  |  |  |  |  |
| AMT-L [CVPR'23] | 12.9M |  |  |  |  |  |
| Ours-L | 待统一脚本统计 |  |  |  |  |  |

执行原则：

- 所有 baseline 使用官方 pretrained，不在 synthetic 数据上 finetune。
- RGB VFI baseline 输入使用 texture endpoint 复制成 3 通道，输出转回 1 通道评估。
- Ours-L 使用 T2exture 训练方案，接收 passive context。
- 所有方法必须跑同一个 test split，并保存完整 `pred/err/metrics/vis`。

当前状态：

| Method | 状态 |
|---|---|
| IFRNet | 权重已存在，wrapper / eval runner 未完成 |
| SGM-VFI | 需要重评到新 split 和新指标 |
| BiM-VFI | 需要重评到新 split 和新指标 |
| GIMM-VFI-F | 官方 repo 已下载，权重已存在，wrapper / eval runner 未完成 |
| AMT-L vanilla | 已完成：PSNR 25.3512 / SSIM 0.9401 / Edge-FI@2px 0.9448 / IE 3.2938 / NIE 0.0129 |
| Ours-L | 需要按 `dataset_roi` 正式配置训练 / 评估 / 可视化，并同步指标 |

表 1 正式输出路径：

| Method | Output root |
|---|---|
| IFRNet | `outputs/final/table01_prior/ifrnet` |
| SGM-VFI | `outputs/final/table01_prior/sgm-vfi` |
| BiM-VFI | `outputs/final/table01_prior/bim-vfi` |
| GIMM-VFI-F | `outputs/final/table01_prior/gimm-vfi-f` |
| AMT-L vanilla | `outputs/final/table01_prior/amt-l-vanilla` |
| Ours-L | `outputs/final/table01_prior/ours-l` |

### 表 2：Simulated Dataset Finetuning，放 5.3.1

| Model | Setting | Train/Total Params | PSNR | SSIM | Edge-FI@2px | IE | NIE |
|---|---|---:|---:|---:|---:|---:|---:|
| AMT-S | vanilla |  | 24.3697 | 0.9343 | 0.9486 | 3.7356 | 0.0146 |
| AMT-S | T2exture |  |  |  |  |  |  |
| AMT-L | vanilla |  | 25.3512 | 0.9401 | 0.9448 | 3.2938 | 0.0129 |
| AMT-L | T2exture |  |  |  |  |  |  |
| AMT-G | vanilla |  | 25.4356 | 0.9401 | 0.9431 | 3.2880 | 0.0129 |
| AMT-G | T2exture |  |  |  |  |  |  |

执行原则：

- `vanilla`：官方 AMT pretrained，直接输入 active texture endpoints，不使用 passive。
- `T2exture`：adapter + passive geometry/fusion，在 synthetic train split 上训练。
- AMT-S/L/G 的 T2exture 都使用 `10000 + 5000` iterations。
- AMT-G 第二阶段解冻后端 selected modules 时使用 `update2_low + update2_high`。

### 表 3：Passive Context Reference，放 5.3.3

| Passive refs per side | Total passive refs | PSNR | SSIM | Edge-FI@2px | IE | NIE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 |  |  |  |  |  |
| 1 | 2 |  |  |  |  |  |
| 2 default | 4 |  |  |  |  |  |
| 3 | 6 |  |  |  |  |  |
| 4 | 8 |  |  |  |  |  |
| 5 | 10 |  |  |  |  |  |

执行原则：

- 固定 backbone：AMT-L。
- 固定 loss、schedule、split、active stride。
- 只改变 passive reference 数量。
- `0` 是 no-passive ablation，需要先补代码。

### 表 4：Active Frame Sparsity，放 5.3.2

| Passive frames between active anchors | Active stride | PSNR | SSIM | Edge-FI@2px | IE | NIE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 |  |  |  |  |  |
| 2 | 3 |  |  |  |  |  |
| 3 | 4 |  |  |  |  |  |
| 4 | 5 |  |  |  |  |  |
| 5 | 6 |  |  |  |  |  |
| 6 | 7 |  |  |  |  |  |
| 7 | 8 |  |  |  |  |  |
| 8 | 9 |  |  |  |  |  |
| 9 default | 10 |  |  |  |  |  |
| 10 | 11 |  |  |  |  |  |

执行原则：

- 固定 backbone：AMT-L。
- 固定 passive context：默认 total passive refs = 4。
- 固定 loss 和训练 schedule。
- 只改变 active stride。
- 全部都是 active frames 属于 oracle upper bound，不放进同一条折线。

每个 stride 需要的 pseudo-flow set：

| Active stride | Flow set | 生成状态 |
|---:|---|---|
| 2 | `flow/s02` | 未生成 |
| 3 | `flow/s03` | 未生成 |
| 4 | `flow/s04` | 未生成 |
| 5 | `flow/s05` | 未生成 |
| 6 | `flow/s06` | 未生成 |
| 7 | `flow/s07` | 未生成 |
| 8 | `flow/s08` | 未生成 |
| 9 | `flow/s09` | 未生成 |
| 10 | `flow/s10` | ROI 已由 source 裁剪生成，覆盖检查通过 |
| 11 | `flow/s11` | 未生成 |

生成命令模板：

```powershell
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset_roi --flow-set s05 --active-stride 5 --splits train valid test --device cuda
```

已确认要生成全套 Table 4 pseudo-flow set。正式执行时按 `s02, s03, ..., s11` 顺序生成；`s10` 不重新跑 LiteFlowNet，直接由已有 `dataset/flow/s10` 裁剪得到 `dataset_roi/flow/s10`，再做覆盖检查。不覆盖原始 `dataset/flow/s10` 全分辨率缓存。所有正式 set 都保存在 `dataset_roi/flow/<flow_set>`，不写回 `dataset_roi/sim/<scene>`，也不写回原始 `dataset/sim/<scene>`。

### 表 5：Practical Safety and Deployability，放 5.4

| Model | Setting | Train/Total Params | PSNR | Latency | FLOPs |
|---|---|---:|---:|---:|---:|
| AMT-S | vanilla |  |  |  |  |
| AMT-S | T2exture |  |  |  |  |
| AMT-L | vanilla |  |  |  |  |
| AMT-L | T2exture |  |  |  |  |
| AMT-G | vanilla |  |  |  |  |
| AMT-G | T2exture |  |  |  |  |

执行原则：

- 和表 2 使用同一批 checkpoint。
- 额外统计 latency / FLOPs。
- latency 必须记录 GPU 型号、batch size、输入分辨率、warmup 次数和重复次数。
- FLOPs 必须使用同一 profiling 函数。

## 6. 旧实验怎么用

不能直接填最终论文主表的原因：

- split 不是当前 20 / 4 / 8。
- 指标缺 `Edge-FI@2px / IE / NIE`。
- Ours 训练 schedule 不是当前 `10000 + 5000` iteration 协议。
- 部分 loss ablation 是 Ours-v0 探索，不是最终固定协议。
- context ablation 用的是 radius，不是准确的 total passive refs。

仍然有价值的结论：

- Ours-v0 在旧 split 上明显优于 AMT-L vanilla，可作为 sanity check。
- 去掉 `L_edge` 后旧实验没有变差，因此 edge loss 不作为主贡献。
- `L_temp` 和 `L_flow` 在旧 Ours-v0 上没有稳定大幅提升，因此正式表中谨慎解释。
- no-passive 旧实验下降明显，可以支持 passive guidance 的必要性，但必须用新 split 重跑。

## 7. 推荐执行顺序

第一阶段先跑能支撑主结论的实验：

1. [done] 写 ROI preview / manifest 脚本，只读分析 `dataset/sim`，为每个 scene 生成固定 crop box。
2. [done] 输出 `roi_preview/<scene>.png` 和 `roi_manifest.json`，后续可继续人工抽查关键 scene。
3. [done] 生成 `dataset_roi`：复制 20 / 4 / 8 split，同步裁剪 texture / passive，并从 `dataset/flow/s10` 裁剪得到 `dataset_roi/flow/s10`。
4. [done] 验证 `dataset_roi`：所有 scene 尺寸一致，train/valid/test sample 数一致，`flow/s10` 覆盖完整。
5. [done] 正式训练配置已固定为 `crop_size: 384`，所有正式命令使用 `--data-root dataset_roi`；验证/测试保持完整 `640x960` ROI。
6. [done] 补英文 runbook 和一键 preflight 脚本，正式长跑前先跑 `preflight_command`。
7. [done] 跑 smoke：确认 AMT-S/L/G wrapper、train/eval/vis 入口都能在 `dataset_roi` 上正常工作。
8. [done] 跑 AMT-L vanilla 新 split 评估，导出完整 artifact。
9. 训练 Ours-L，保存 `best.pt` / `last.pt` / `config.json`。
10. 评估 Ours-L，导出 `pred/err/metrics/vis`。
11. [code done] 补 IFRNet wrapper 并跑表 1。
   - 当前状态：官方 IFRNet VFI checkpoint 已补，等待 GPU 空闲后运行表 1 IFRNet 正式评估；`pretrained/LiteFlowNet.pth` 只用于 pseudo-flow / LiteFlowNet，不可作为 IFRNet baseline。
12. 重评 SGM-VFI 和 BiM-VFI 到新 split / 新指标。
13. [code downloaded] `third_party/GIMM-VFI` 已存在；仍需补 GIMM-VFI-F wrapper 并跑表 1。

第二阶段跑模型规模消融和部署表：

14. [done] 跑 AMT-S / AMT-G vanilla。
15. 训练 AMT-S / AMT-G T2exture。
16. 评估 AMT-S/L/G 的 vanilla 和 T2exture。
17. 补 latency / FLOPs profiling，填表 5。

第三阶段再跑额外消融：

18. 补 no-passive 和 variable passive context，跑表 3。
19. 补 variable active stride dataset。
20. 在 `dataset_roi` 下生成 Table 4 的全套 pseudo-flow set：`flow/s02...flow/s11`，其中 `flow/s10` 已由裁剪复用得到，只做覆盖检查。
21. 跑 Table 4 active frame sparsity。
22. 暂时不跑 `w/o L_flow`；loss 相关探索等主表和 Table 4 稳定后再决定是否追加。

## 8. 当前不建议做的事

- 不把 real transfer 混进当前 synthetic-only 表格。
- 不把旧 split 的结果直接填进论文主表。
- 不把 `L_edge / L_temp / L_flow` 的旧探索结果说成最终主贡献。
- 当前不做 `w/o L_flow`。
- 不在 AMT-G 上直接套 AMT-L 的 `update2` 模块名。
- 不把 generated pseudo-flow 写回 `dataset/sim/<scene>`。
- `dataset_roi` 生成确认后，不再用原始 `dataset` 跑正式表。
- 不覆盖原始 `dataset/flow/s10` 全分辨率缓存；只生成裁剪后的 `dataset_roi/flow/s10`。
- 不逐帧动态裁剪；每个 scene 必须使用固定 crop box。
- 不 resize ROI 数据；只做 crop / pad，避免 flow 数值缩放误差。
- 不再投入 InterpAny、LDF-VFI、EDEN。
- 不提交 `third_party/`、`dataset/`、`pretrained/`、`outputs/`。
