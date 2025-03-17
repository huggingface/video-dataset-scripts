import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List
import json


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


def load_vlm_model(gemma_ckpt, use_fa2=False):
    if not torch.cuda.is_available():
        raise ValueError("Must use a CUDA device.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("Must use a CUDA device with at least 8.0 compute capability to use Bfloat16.")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_fa2 else None,
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(gemma_ckpt)
    return model, processor


def create_messages(keyframes: List[Image.Image], effect: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": f"Does the video have actions of {effect}?"}]},
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
def run_vlm(model, processor, messages_list):
    inputs = processor.apply_chat_template(
        messages_list,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        # To enable batched inference with Gemma 3 we enable padding and
        # specify the side for padding.
        padding=True,
        padding_side="left",
    ).to(model.device)
    generations = model.generate(**inputs, max_new_tokens=500, do_sample=False)

    num_items = inputs["input_ids"].shape[0]
    decoded_batch = []
    for idx in range(num_items):
        input_len = inputs["input_ids"].shape[-1]
        generation = generations[idx, input_len:]
        decoded_batch.append(processor.decode(generation, skip_special_tokens=True))
    return decoded_batch
