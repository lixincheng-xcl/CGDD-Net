"""Memory-bounded overlap averaging, adapted from VesselSeg's patch workflow."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
import torch
from .metrics import binary_metrics, summarize


@torch.inference_mode()
def sliding_window(model, image, patch_size=96, stride=16, batch_size=8, device="cpu"):
    if patch_size < 16 or not 1 <= stride <= patch_size or batch_size < 1:
        raise ValueError("Require patch_size >= 16, 1 <= stride <= patch_size, batch_size >= 1")
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Expected a finite nonempty C,H,W image")
    _, h, w = array.shape
    padded_h = patch_size + max(0, (h-patch_size+stride-1)//stride)*stride
    padded_w = patch_size + max(0, (w-patch_size+stride-1)//stride)*stride
    mode = "reflect" if min(h, w) > 1 else "edge"
    array = np.pad(array, ((0,0), (0,padded_h-h), (0,padded_w-w)), mode=mode)
    total, count = np.zeros((padded_h,padded_w), np.float32), np.zeros((padded_h,padded_w), np.float32)
    locations = [(y,x) for y in range(0,padded_h-patch_size+1,stride)
                 for x in range(0,padded_w-patch_size+1,stride)]
    was_training = model.training
    model.eval()
    try:
        for start in range(0,len(locations),batch_size):
            coords = locations[start:start+batch_size]
            patches = torch.from_numpy(np.stack([array[:,y:y+patch_size,x:x+patch_size] for y,x in coords])).to(device)
            output = model(patches)
            if output.shape != (len(coords), 1, patch_size, patch_size):
                raise ValueError(f"Unexpected model output shape: {output.shape}")
            probs = output.sigmoid().cpu().numpy()[:,0]
            for (y,x), p in zip(coords,probs):
                total[y:y+patch_size,x:x+patch_size] += p
                count[y:y+patch_size,x:x+patch_size] += 1
    finally:
        model.train(was_training)
    if not np.all(count > 0):
        raise RuntimeError("Sliding-window grid left uncovered pixels")
    return (total/count)[:h,:w]


def evaluate_records(model, records, preprocessing=None, inference=None, device="cpu", output=None):
    from .data import load_case
    options = dict(inference or {})
    threshold = options.pop("threshold",0.5)
    rows = []
    output = Path(output) if output else None
    if output:
        output.mkdir(parents=True,exist_ok=True)
        (output/"predictions").mkdir(exist_ok=True)
    for record in records:
        image, label, fov = load_case(record, preprocessing=preprocessing)
        prob = sliding_window(model,image,device=device,**options)
        with Image.open(record["image"]) as original:
            original_width, original_height = original.size
        row = {"id": record["id"], "original_height": original_height, "original_width": original_width,
               "evaluated_height": prob.shape[0], "evaluated_width": prob.shape[1], **binary_metrics(prob,np.asarray(label)[0],np.asarray(fov)[0],threshold)}
        rows.append(row)
        if output:
            # Name by row index to avoid unsafe path characters or stem collisions.
            name = f"{len(rows):04d}"
            np.savez_compressed(output/"predictions"/(name+".npz"),probability=prob)
            Image.fromarray(((prob>=threshold)*255).astype(np.uint8)).save(output/"predictions"/(name+".png"))
            row["prediction"] = "predictions/"+name+".npz"
    summary = summarize(rows)
    summary["threshold"] = threshold
    summary["dispersion_note"] = "std_between_images is not seed-to-seed training variability"
    if output:
        with (output/"per_image.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary
