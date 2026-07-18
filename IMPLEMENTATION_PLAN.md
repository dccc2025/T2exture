# TTexture Edge-Conditioned AMT Implementation Plan

**Goal:** Fine-tune a frozen AMT-L backbone with a zero-initialized, passive-edge adapter that predicts nine texture frames between texture keyframes spaced ten frames apart.

**Architecture:** Copy only AMT runtime modules into `TTexture`; add a three-level Sobel-edge encoder and zero-initialized residual injections at AMT intermediate scales. Train adapters first, then optionally unfreeze the final multi-field decoder and combiner. All splits are scene-disjoint.

**Constraints:** Preserve `/essfs10/daicheng/AMT`; use only `TT_sim/train_valid_test_splits`; each sample uses texture frames `1,11,...,171` as endpoints and passive frames between them; save reproducible configuration, checkpoints, metrics, and curves.

## Tasks

1. Copy AMT runtime code (`networks`, `utils`, `losses`) and the AMT-L checkpoint into `TTexture`; exclude assets, demos, benchmark datasets, docs, and original training code.
2. Add a dataset test that asserts scene-disjoint splits and constructs exactly nine targets (`002`--`010`) from endpoint texture frames `001` and `011`, with matching passive controls.
3. Implement and test the edge-control adapter: zero-initialized projections must yield the unmodified AMT output before learning; frozen backbone parameters must receive no gradients in phase 1.
4. Implement deterministic training, validation, checkpointing, and phase transition. Optimize Charbonnier, contrast-structure, edge-weighted gradient, and temporal-difference losses.
5. Implement held-out test inference over all test scenes, reporting PSNR, SSIM, MAE, and Alex-LPIPS only on interpolated frames; render the training curve.
6. Verify artifact counts, split isolation, output finiteness, checkpoint provenance, and reported metrics.
