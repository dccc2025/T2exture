# Pseudo-Flow Generation

T2exture follows AMT's storage pattern: raw frames and pseudo-flow caches are
kept separate.

```text
dataset/
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
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset --flow-set s10 --active-stride 10 --splits train valid test --device cuda
```

Generate another active-sparsity set:

```powershell
conda run -n gflow python -B -m flow_generation.generate_liteflownet_flow --data-root dataset --flow-set s05 --active-stride 5 --splits train valid test --device cuda
```

Legacy `dataset/sim/<scene>/flow/*.npz` files are still readable for the default
`s10` set, but new generated caches should go under `dataset/flow/<flow_set>`.
