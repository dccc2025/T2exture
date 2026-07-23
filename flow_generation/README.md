# Pseudo-Flow Generation

T2exture follows AMT's storage pattern: raw frames and pseudo-flow caches are
kept separate.

```text
dataset_roi/
  sim/<scene>/texture/*.npy
  sim/<scene>/passive/*.npy
  flow/<flow_set>/<scene>/<left>_<target>_<right>.npz
```

Each `.npz` stores:

- `flow0`: target-to-left flow, shape `[2, H, W]`
- `flow1`: target-to-right flow, shape `[2, H, W]`

Default training uses `flow/s10`, where `s10` means active anchors are ten
frames apart.

Generate the default set:

```powershell
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset_roi --active-stride 10 --splits train valid test --device cuda
```

Generate another active-sparsity set:

```powershell
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset_roi --active-stride 5 --splits train valid test --device cuda
```

When `--flow-set` is omitted, the generator writes to `flow/sXX` based on
`--active-stride`, for example `active_stride=5` writes `flow/s05`.

Legacy `dataset/sim/<scene>/flow/*.npz` files are still readable for source
inspection only. Formal runs should read cropped caches from
`dataset_roi/flow/<flow_set>`.
