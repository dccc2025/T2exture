from pathlib import Path


def test_sequence_dataset_exposes_passive_endpoint_geometry():
    from tt_data import TextureSequenceDataset

    root = Path('/essfs10/daicheng/TT_sim')
    dataset = TextureSequenceDataset(root, root / 'train_valid_test_splits' / 'train.txt', crop_size=None)
    item = dataset[0]
    assert item['p0'].shape == (1, 768, 1024)
    assert item['p1'].shape == (1, 768, 1024)
    assert item['left_id'] == 1
