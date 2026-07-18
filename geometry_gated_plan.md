# Geometry-Gated Two-Endpoint AMT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve two-endpoint texture interpolation by using passive thermal frames only for correspondence and learned fusion gates, while keeping texture synthesis sourced from the two observed texture endpoints.

**Architecture:** A one-by-one thermal adapter maps each single-channel texture endpoint into AMT-L's three-channel input space and is initialized to exactly reproduce channel replication. A passive triplet `(P_left, P_right, P_target)` is processed by a small geometry branch which predicts two endpoint warp fields and two signed per-pixel fusion gates. AMT remains the base prediction; the output is its gated correction from the two geometry-warped texture endpoints, so passive data never supplies texture values.

**Tech Stack:** PyTorch, AMT-L, CUDA, OpenCV/ffmpeg.

## Global Constraints

- Retain the two-endpoint interpolation task; do not add multi-anchor retrieval or sequence generation.
- Train and evaluate on `/essfs10/daicheng/TT_sim` using the established split files.
- Run experiments only with `CUDA_VISIBLE_DEVICES=3`.
- Preserve `/essfs10/daicheng/AMT`; modify only `/essfs10/daicheng/TTexture`.

---

### Task 1: Expose endpoint passive frames and define the regression contract

**Files:**
- Modify: `tt_data.py`
- Create: `tests/test_geometry_gated_contracts.py`

- [ ] **Step 1: Write the failing contract test**

```python
def test_geometry_gated_model_starts_as_pretrained_amt_with_replicated_input():
    model = GeometryGatedAMT().eval()
    assert torch.allclose(model.input_adapter(gray), gray.repeat(1, 3, 1, 1))
    assert torch.allclose(model(gray0, gray1, time, p0, p1, pt), amt_prediction)
```

- [ ] **Step 2: Run the test and confirm it fails because `GeometryGatedAMT` is absent.**

- [ ] **Step 3: Add `p0` and `p1` to interval/sequence dataset examples.**

- [ ] **Step 4: Re-run the dataset and model contracts.**

### Task 2: Implement thermal adaptation and passive-only geometry-gated fusion

**Files:**
- Modify: `tt_model.py`

- [ ] **Step 1: Implement `ThermalInputAdapter` with a 1x1 `1 -> 3` convolution initialized to channel replication.**
- [ ] **Step 2: Implement `PassiveGeometryFusion`, which consumes only `p0`, `p1`, and `pt`, predicts two zero-initialized flow corrections and two zero-initialized gates, and warps endpoint texture images with `grid_sample`.**
- [ ] **Step 3: Implement `GeometryGatedAMT` with**

```python
base = backbone(input_adapter(x0), input_adapter(x1), time)
warp0, warp1, gates = geometry(p0, p1, pt, x0, x1)
return base + gates[:, :1] * (warp0 - base) + gates[:, 1:] * (warp1 - base)
```

- [ ] **Step 4: Run the new contract test and existing dataset contracts.**

### Task 3: Train/evaluate the controlled geometry-gated variant

**Files:**
- Modify: `train_ttexture.py`, `evaluate_ttexture.py`, `make_cow_videos.py`

- [ ] **Step 1: Pass endpoint passive tensors through training, validation, evaluation and cow-video inference.**
- [ ] **Step 2: Retain the existing Charbonnier, CSS, edge, and view-temporal losses so the architecture is the primary controlled variable.**
- [ ] **Step 3: Run two-stage training and evaluate all 1,530 test frames against the existing AMT-L baseline.**
- [ ] **Step 4: Render a 180-frame, 30 fps, six-second cow prediction video using 18 original texture anchors and 162 inferred frames.**

### Task 4: Verify outputs

- [ ] **Step 1: Check the 10-epoch history, finite metrics, checkpoint, and loss curve.**
- [ ] **Step 2: Check both cow videos have 180 frames, 30 fps, six seconds, and the predicted sequence has 18 anchors plus 162 inferred frames.**
