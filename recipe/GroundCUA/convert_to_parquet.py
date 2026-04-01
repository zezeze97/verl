from pathlib import Path

import pandas as pd

df = pd.read_parquet("input.parquet")

def convert_images(images):
    out = []
    for item in images or []:
        path = item.get("path")
        if path:
            out.append({"image": Path(path).resolve().as_uri()})
        elif item.get("bytes") is not None:
            out.append({"bytes": item["bytes"]})
    return out

df["images"] = df["images"].apply(convert_images)
df.to_parquet("output.parquet", index=False)
