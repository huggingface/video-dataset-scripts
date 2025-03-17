import pandas as pd
import pathlib
from argparse import ArgumentParser
from tqdm import tqdm
from modules import separate_key_frames_from_row, load_vlm_model, run_vlm, recover_json_from_output, create_messages

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--parquet-path", type=str, required=True)
parser.add_argument(
    "--effect",
    type=str,
    required=True,
    help="Type of effect the VLM should look for. Don't forget to modify the `SYSTEM_PROMPT` accordingly in modules/gemma3.py.",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="google/gemma-3-27b-it",
    choices=["google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"],
)
parser.add_argument("--use_fa2", action="store_true", help="If using Flash Attention 2.")
parser.add_argument("--parquet-out-path", type=str, required=True)
parser.add_argument("--chunk_size", type=int, default=4, help="Infer on batches of rows")
args = parser.parse_args()
path = pathlib.Path(args.path)

effect = args.effect
model_id = args.model_id
use_fa2 = args.use_fa2
parquet_path = pathlib.Path(args.parquet_path)
parquet_out_path = pathlib.Path(args.parquet_out_path)
chunk_size = args.chunk_size

model, processor = load_vlm_model(model_id, use_fa2=use_fa2)
df = pd.read_parquet(parquet_path)

data = []
messages_list = []
answer_list = []
with tqdm() as pbar:
    for idx, row in df.iterrows():
        pbar.set_description(row["file"])
        key_frames, first, mid, last = separate_key_frames_from_row(path, row)
        pbar.set_postfix_str(f"{len(key_frames)} key frames")

        messages = create_messages(key_frames, effect=effect)

        if idx % chunk_size == 0 and idx != 0:
            answer_list = run_vlm(model, processor, messages_list)            
            answer_list = [recover_json_from_output(answer) for answer in answer_list]

            row = [{f"{effect}": answer[effect], "confidence": float(answer["confidence"])} for answer in answer_list]
            data.extend(row)
            pbar.update()
            messages_list = list()
        else:
            messages_list.append(messages)

shot_categorized_df = pd.DataFrame(data)

print(shot_categorized_df)

df = df.join(shot_categorized_df)

print(df)

df.to_parquet(parquet_out_path)
