import torch


def test_geometry_gated_model_starts_as_pretrained_amt_with_replicated_input():
    from tt_model import GeometryGatedAMT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(7)
    model = GeometryGatedAMT().to(device).eval()
    gray0 = torch.rand(1, 1, 128, 128, device=device)
    gray1 = torch.rand(1, 1, 128, 128, device=device)
    p0, p1, pt = (torch.rand(1, 1, 128, 128, device=device) for _ in range(3))
    time = torch.full((1, 1, 1, 1), 0.5, device=device)
    with torch.no_grad():
        adapted = model.input_adapter(gray0)
        reference = model.backbone(adapted, model.input_adapter(gray1), time, eval=True)['imgt_pred']
        predicted = model(gray0, gray1, time, p0, p1, pt)
        flows, gates = model.geometry(p0, p1, pt)
    assert torch.allclose(adapted, gray0.repeat(1, 3, 1, 1), atol=1e-7, rtol=0)
    assert torch.allclose(flows, torch.zeros_like(flows), atol=1e-7, rtol=0)
    assert torch.allclose(gates, torch.zeros_like(gates), atol=1e-7, rtol=0)
    assert torch.allclose(predicted, reference, atol=1e-6, rtol=0)
    assert torch.isfinite(predicted).all()
