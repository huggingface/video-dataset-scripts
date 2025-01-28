import pandas as pd
import pathlib
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from modules import run, load_florence

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


load_florence(
    hf_hub_or_path="microsoft/Florence-2-large",
    device=device,
    dtype=dtype,
)


df = pd.read_parquet(parquet_path)

task_prompt = [
    "<DENSE_REGION_CAPTION>",
    "<OCR_WITH_REGION>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
]

data = []
with tqdm() as pbar:
    for _, row in df.iterrows():
        pbar.set_description(row["file"])
        key_frames = [Image.open(path.joinpath(key_frame)) for key_frame in row["frames"]]
        pbar.set_postfix_str(f"{len(key_frames)} key frames")
        first = key_frames[0]
        mid = None
        last = None
        if len(key_frames) == 2:
            last = key_frames[1]
        elif len(key_frames) > 2:
            mid = key_frames[len(key_frames) // 2]
            last = key_frames[-1]
        frames = [first]
        first = run(first, task_prompt=task_prompt)
        caption = [first["<CAPTION>"]]
        detailed_caption = [first["<DETAILED_CAPTION>"]]
        region_caption = [first["<DENSE_REGION_CAPTION>"]]
        ocr_region = [first["<OCR_WITH_REGION>"]]
        if mid:
            frames.append(mid)
            mid = run(mid, task_prompt=task_prompt)
            caption.append(mid["<CAPTION>"])
            detailed_caption.append(mid["<DETAILED_CAPTION>"])
            region_caption.append(mid["<DENSE_REGION_CAPTION>"])
            ocr_region.append(mid["<OCR_WITH_REGION>"])
        if last:
            frames.append(last)
            last = run(last, task_prompt=task_prompt)
            caption.append(last["<CAPTION>"])
            detailed_caption.append(last["<DETAILED_CAPTION>"])
            region_caption.append(last["<DENSE_REGION_CAPTION>"])
            ocr_region.append(last["<OCR_WITH_REGION>"])
        row = {
            "caption": caption,
            "detailed_caption": detailed_caption,
            "region_caption": region_caption,
            "ocr": ocr_region,
        }
        data.append(row)
        pbar.update()

caption_df = pd.DataFrame(data)

print(caption_df)

df = df.join(caption_df)

print(df)

df.to_parquet(parquet_out_path)
