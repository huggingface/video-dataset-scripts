import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import snapshot_download
from tqdm import tqdm
import pandas as pd
import json
import glob
import av

GEMMA_CKPT = "google/gemma-3-27b-it"  # smol: "google/gemma-3-4b-it"

SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to determine if the key frames of a video
has actions of 'squishing'. You will be provided with the keyframes of a video as inputs.

To classify if the provided input has an effect of squishing, look out for a pair of hands
and if they squeezing any objects into some compressible form. This must be followed
at all times.

Please return the answer in the form of a dictionary with the following
keys: `squishing` and `confidence`. `squishing` should be a binary answer yes/no.
`confidence` should be within [0,1]. Don't include any reasoning or any extra stuff
in the response.
"""

BATCH_SIZE = 16

USE_KEYFRAMES_ONLY = True


def get_key_frames(path):
    frames = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    if USE_KEYFRAMES_ONLY:
        stream.codec_context.skip_frame = "NONKEY"
    for _, frame in enumerate(container.decode(stream)):
        frames.append(frame.to_image())
    container.close()
    return frames


def load_model():
    ckpt = GEMMA_CKPT
    model = Gemma3ForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor


def create_messages(keyframes):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"Does the video have actions of squishing?"}]},
    ]
    for i, frame in enumerate(keyframes):
        messages[1]["content"].append({"type": "text", "text": f"Key frame {i}:"})
        messages[1]["content"].append({"type": "image", "url": frame})

    return messages


def recover_json_from_output(output: str):
    start = output.find("{")
    end = output.rfind("}") + 1
    json_part = output[start:end]
    return json.loads(json_part)


@torch.no_grad()
@torch.inference_mode()
def main():
    model, processor = load_model()
    repo_id = "diffusers-internal-dev/squish-unfiltered"
    dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir="/fsx/sayak/.cache")
    all_video_paths = glob.glob(f"{dataset_path}/*.mp4")

    answers = {}
    for video_path in tqdm(all_video_paths):
        keyframes = get_key_frames(video_path)
        messages = create_messages(keyframes)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

        answers[video_path.split("/")[-1]] = recover_json_from_output(decoded)

    return answers


if __name__ == "__main__":
    answers = main()
    df_dict = {
        "path": list(answers.keys()),
        "squishing": [str(sample["squishing"]) for _, sample in answers.items()],
        "confidence": [float(sample["confidence"]) for _, sample in answers.items()],
    }
    df = pd.DataFrame(df_dict)
    df.to_parquet("gemma3.parquet", index=False)
