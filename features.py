from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def extract_features(block_name, masks, channels_dict):
    props = regionprops(masks)
    rows = []

    for p in props:
        cid = p.label
        cy, cx = p.centroid

        row = {
            "block": block_name,
            "cell_id": int(cid),
            "global_cell_id": "%s_%s" % (block_name, cid),
            "area": float(p.area),
            "eccentricity": float(p.eccentricity),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
        }

        rr = p.coords[:, 0]
        cc = p.coords[:, 1]

        for ch_name, img in channels_dict.items():
            vals = img[rr, cc].astype(np.float64)
            row["%s_mean" % ch_name] = float(vals.mean())
            row["%s_std" % ch_name] = float(vals.std())
            row["%s_max" % ch_name] = float(vals.max())

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fiji_stitcher.config import load_config
    from loader import load_block
    from segmentation import segment_nuclei

    config = load_config()
    block_name = "A4"
    imgs = load_block(
        block_name=block_name,
        crop_root=config["CROP_OUTPUT_DIR"],
        channels=config.get("LOADER", {}).get("CHANNELS", ["DAPI", "HER2", "PR", "ER"]),
        do_preprocess=bool(config.get("LOADER", {}).get("DO_PREPROCESS", False)),
        loader_cfg=config.get("LOADER", {}),
    )

    masks = segment_nuclei(imgs["DAPI"], seg_cfg=config.get("SEGMENTATION", {}))
    df = extract_features(block_name, masks, imgs)

    out_dir = Path(config.get("FEATURE_OUTPUT_DIR", config["BATCH_OUTPUT_DIR"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / ("%s_cell_features_with_labels.csv" % block_name)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved: %s" % out_csv)
