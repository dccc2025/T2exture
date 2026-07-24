# Pretrained Checkpoints

This directory stores local pretrained weights used by the T2exture synthetic experiments.
Only this README is intended to be tracked. Checkpoint files remain ignored.

## Current local layout

AMT
- Current paths:
  - pretrained/amt-s.pth
  - pretrained/amt-l.pth
  - pretrained/amt-g.pth
- Purpose: vanilla AMT-S/L/G evaluation and T2exture AMT backbone initialization.
- Official code: https://github.com/MCG-NKU/AMT
- Known download page: https://huggingface.co/lalala125/AMT
- Note: these files are still at the pretrained root because current scripts and running jobs reference them directly. Move them to pretrained/AMT/ only after updating all commands and configs.

LiteFlowNet
- Current paths:
  - pretrained/LiteFlowNet.pth
  - pretrained/liteflownet/hub/checkpoints/liteflownet-default
- Purpose: pseudo-flow teacher/cache generation, not a Table 1 VFI baseline.
- Reference implementation used locally: https://github.com/sniklaus/pytorch-liteflownet

IFRNet
- Current paths:
  - pretrained/IFRNet/IFRNet_Vimeo90K.pth
  - pretrained/IFRNet/IFRNet_GoPro.pth
- Purpose: Table 1 IFRNet baseline.
- Default for formal comparison: pretrained/IFRNet/IFRNet_Vimeo90K.pth
- Official code: https://github.com/ltkong218/IFRNet
- Weight download: https://www.dropbox.com/scl/fo/gvfjc8bq259l4cre2ai0k/AIxkWTcEOcvIIYe7RDlZpag?rlkey=x4lxph520gbt0tjy839gmwoc0&e=3&dl=0
- Note: use the Vimeo90K checkpoint for generic VFI comparison. The GoPro checkpoint is motion-blur oriented and should not replace it unless explicitly documented.

GIMM-VFI-F
- Current paths:
  - pretrained/GIMM-VFI-F/flowformer_sintel.pth
  - pretrained/GIMM-VFI-F/gimm.pt
  - pretrained/GIMM-VFI-F/gimmvfi_f_arb.pt
- Purpose: Table 1 GIMM-VFI-F baseline.
- Official code: https://github.com/GSeanCDAT/GIMM-VFI
- Required components: FlowFormer/RAFT-style flow backbone, GIMM module, and synthesis network checkpoint.

SGM-VFI
- Current paths:
  - pretrained/sgm-vfi/pretrained/gmflow_sintel-0c07dcb3.pth
  - pretrained/sgm-vfi/log/...
- Purpose: Table 1 SGM-VFI baseline once the local runner is wired.
- Official code: https://github.com/MCG-NJU/SGM-VFI

BiM-VFI
- Current path:
  - pretrained/bim-vfi/bim_vfi.pth
- Purpose: Table 1 BiM-VFI baseline once the local runner is wired.
- Official code: https://github.com/KAIST-VICLab/BiM-VFI

Unused or deprioritized baselines
- pretrained/eden/ and pretrained/ldf-vfi/ are kept only as local downloads unless a future experiment explicitly re-enables them.
- InterpAny, LDF-VFI, and EDEN are not part of the current formal Table 1 plan.

## Sanity notes

- Do not commit checkpoint files.
- Keep baseline weights under method-specific subfolders when possible.
- If a path is changed, update docs/formal_experiment_runbook.yaml, scripts, and command examples in the same change.
