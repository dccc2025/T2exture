from pathlib import Path

import torch


DATA_ROOT = Path('/essfs10/daicheng/TT_sim')
TRAIN_SPLIT = DATA_ROOT / 'train_valid_test_splits' / 'train.txt'


def test_train_dataset_uses_scene_disjoint_ten_frame_intervals():
    from tt_data import TextureIntervalDataset

    dataset = TextureIntervalDataset(DATA_ROOT, TRAIN_SPLIT, crop_size=None)
    assert len(dataset.scenes) == len(TRAIN_SPLIT.read_text().splitlines()) == 20
    assert len(dataset) == 20 * 17 * 9
    sample = dataset[0]
    assert sample['scene'] == 'adjustable_wrench'
    assert sample['frame_ids'] == (1, 11, 2)
    assert sample['x0'].shape == (1, 768, 1024)
    assert sample['x1'].shape == (1, 768, 1024)
    assert sample['p'].shape == (1, 768, 1024)
    assert sample['target'].shape == (1, 768, 1024)
    assert torch.isclose(sample['time'], torch.tensor(0.1))


def test_zero_initialized_edge_adapter_preserves_frozen_amt_prediction():
    from tt_model import EdgeConditionedAMT

    torch.manual_seed(0)
    # CUDA_VISIBLE_DEVICES=3 maps the requested physical GPU to cuda:0 here.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeConditionedAMT().to(device).eval()
    x0 = torch.rand(1, 3, 128, 128, device=device)
    x1 = torch.rand(1, 3, 128, 128, device=device)
    p = torch.rand(1, 1, 128, 128, device=device)
    time = torch.full((1, 1, 1, 1), 0.5, device=device)
    with torch.no_grad():
        baseline = model.backbone(x0, x1, time, eval=True)['imgt_pred']
        conditioned = model(x0, x1, time, p)
    assert torch.isfinite(conditioned).all()
    assert torch.allclose(conditioned, baseline, atol=1e-6, rtol=0)
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.adapter_parameters())
