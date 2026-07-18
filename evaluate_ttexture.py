from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from tt_data import TextureSequenceDataset
from tt_model import GeometryGatedAMT


def arguments():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, default=Path('/essfs10/daicheng/TT_sim'))
    p.add_argument('--checkpoint', type=Path, default=None); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lpips-path', type=Path, default=Path('/essfs10/daicheng/AMT/third_party')); p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = arguments(); args.output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = GeometryGatedAMT().to(device)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False); model.load_state_dict(state['model'], strict=True)
    model.eval()
    sys.path.insert(0, str(args.lpips_path)); import lpips
    perceptual = lpips.LPIPS(net='alex').to(device).eval()
    split = args.data_root / 'train_valid_test_splits' / 'test.txt'; dataset = TextureSequenceDataset(args.data_root, split, crop_size=None)
    rows = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            x0, x1 = batch['x0'].to(device), batch['x1'].to(device)
            p0, p1 = batch['p0'].to(device), batch['p1'].to(device)
            for j in range(9):
                passive = batch['p'][:, j].to(device); target = batch['target'][:, j].to(device); time = batch['times'][:, j].to(device).view(1, 1, 1, 1)
                pred = model(x0, x1, time, p0, p1, passive).mean(1, keepdim=True).clamp(0, 1)
                gt, pr = target[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy()
                mse = float(np.mean((gt-pr)**2)); psnr = float('inf') if mse == 0 else float(-10*np.log10(mse))
                ssim = float(structural_similarity(gt, pr, data_range=1.0)); mae = float(np.mean(np.abs(gt-pr)))
                lp = float(perceptual(target.repeat(1,3,1,1)*2-1, pred.repeat(1,3,1,1)*2-1).item())
                rows.append({'scene': batch['scene'][0], 'left_id': int(batch['left_id'][0]), 'target_id': int(batch['left_id'][0])+j+1, 'psnr': psnr, 'ssim': ssim, 'mae': mae, 'lpips': lp})
    with (args.output/'test_per_frame.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'checkpoint':str(args.checkpoint) if args.checkpoint else 'AMT-L zero-adapter baseline','frames':len(rows),'scenes':dataset.scenes,'mean':{k:float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','mae','lpips')},'std':{k:float(np.std([r[k] for r in rows])) for k in ('psnr','ssim','mae','lpips')}}
    (args.output/'test_metrics.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
