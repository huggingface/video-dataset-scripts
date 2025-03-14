import pandas as pd
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from modules import run, load_florence, separate_key_frames_from_row

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument("--parquet-out-path", type=str, required=True)
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--dtype", type=str, required=True)
args = parser.parse_args()
path = pathlib.Path(args.path)
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)
device = args.device
dtype = args.dtype

load_florence(hf_hub_or_path="diffusers/shot-categorizer-v0", device=device, dtype=dtype, check_task_types=False)

df = pd.read_parquet(parquet_path)

task_prompt = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]

data = []
with tqdm() as pbar:
    for _, row in df.iterrows():
        pbar.set_description(row["file"])
        key_frames, first, mid, last = separate_key_frames_from_row(path, row)
        pbar.set_postfix_str(f"{len(key_frames)} key frames")

        frames = [first]
        first = run(first, task_prompt=task_prompt)
        color = [first["<COLOR>"]]
        lighting = [first["<LIGHTING>"]]
        lighting_type = [first["<LIGHTING_TYPE>"]]
        composition = [first["<COMPOSITION>"]]

        if mid:
            frames.append(mid)
            mid = run(mid, task_prompt=task_prompt)
            color.append(mid["<COLOR>"])
            lighting.append(mid["<LIGHTING>"])
            lighting_type.append(mid["<LIGHTING_TYPE>"])
            composition.append(mid["<COMPOSITION>"])

        if last:
            frames.append(last)
            last = run(last, task_prompt=task_prompt)
            color.append(last["<COLOR>"])
            lighting.append(last["<LIGHTING>"])
            lighting_type.append(last["<LIGHTING_TYPE>"])
            composition.append(last["<COMPOSITION>"])

        row = {
            "color": color,
            "lighting": lighting,
            "lighting_type": lighting_type,
            "composition": composition,
        }
        data.append(row)
        pbar.update()

shot_categorized_df = pd.DataFrame(data)

print(shot_categorized_df)

df = df.join(shot_categorized_df)

print(df)

df.to_parquet(parquet_out_path)
