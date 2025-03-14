import os
import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import csv
import pandas as pd
from frames import get_frames


def compute_video_embedding(frames, model, preprocessor, device, dtype):
    """
    Compute video embeddings. `frames` can either be frames of a single video or a list of list of
    frames from multiple videos.
    """
    if not frames:
        return None

    if isinstance(frames[0], list):
        video_embeddings = []
        flat_frames = []
        video_lengths = []

        for video in frames:
            video_lengths.append(len(video))
            flat_frames.extend(video)

        all_input = preprocessor(images=flat_frames, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=dtype):
            embeddings = model(**all_input).pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu()

        # Group the embeddings back by video
        index = 0
        for length in video_lengths:
            video_emb = embeddings[index : index + length].mean(dim=0)
            video_emb = video_emb / video_emb.norm()
            video_embeddings.append(video_emb.numpy())
            index += length

        return video_embeddings
    else:
        all_input = preprocessor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=dtype):
            embeddings = model(**all_input).pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu()

        video_embedding = embeddings.mean(dim=0)
        video_embedding = video_embedding / video_embedding.norm()
        return video_embedding.numpy()


def compute_image_embedding(image_path, model, preprocessor, device, dtype):
    """
    Computes an embedding for a single image.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = preprocessor(image, return_tensors="pt").to(device)
    with torch.no_grad() and torch.autocast(torch.device(device).type, dtype=dtype):
        embedding = model(**image_input).pooler_output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def compute_reference_embedding(ref_path, model, preprocessor, device, dtype):
    """
    Computes the embedding for a reference file (image or video).
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    if ref_path.lower().endswith(video_extensions):
        frames = get_frames(ref_path)
        frames = next(iter(frames))
        frames = [frame.to_image() for frame in frames]
        return compute_video_embedding(frames, model, preprocessor, device, dtype)
    else:
        return compute_image_embedding(ref_path, model, preprocessor, device, dtype)


@torch.no_grad()
@torch.inference_mode()
def main(args):
    # List video files in the folder (supports common video extensions)
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [
        os.path.join(args.videos_folder, f)
        for f in os.listdir(args.videos_folder)
        if f.lower().endswith(video_extensions)
    ]
    print(f"Total video files: {len(video_files)}")
    assert video_files

    # Load model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384", attn_implementation="flash_attention_2"
    ).to(device)
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Process each reference file and average their embeddings.
    ref_embeddings = []
    if os.path.isdir(args.reference):
        allow_extensions = video_extensions + (".png", ".jpg", ".jpeg")
        reference = [
            os.path.join(args.reference, f) for f in os.listdir(args.reference) if f.endswith(allow_extensions)
        ]
    else:
        reference = args.reference.split(",")

    assert reference

    for ref in reference:
        emb = compute_reference_embedding(ref, model, preprocessor, device, dtype)
        if emb is not None:
            ref_embeddings.append(emb)
        else:
            print(f"Could not compute embedding for reference: {ref}")

    if len(ref_embeddings) == 0:
        print("No valid reference embeddings found!")
        return

    ref_embedding = np.mean(ref_embeddings, axis=0)
    ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)

    results = []
    batch_frames = []  # To collect frames for a batch of videos
    batch_paths = []  # To keep track of corresponding video paths
    pbar = tqdm(video_files, desc="Computing video embeddings.")

    for video_path in pbar:
        pbar.set_postfix_str(f"{video_path}")

        frames_generator = get_frames(video_path)
        try:
            frames_batch = next(iter(frames_generator))
        except StopIteration:
            print(f"Could not extract frames from {video_path}")
            continue

        frames = [frame.to_image() for frame in frames_batch]
        if not frames:
            print(f"Could not extract frames from {video_path}")
            continue

        frames = frames[: args.max_num_frames]
        batch_frames.append(frames)
        batch_paths.append(video_path)

        if len(batch_frames) == args.batch_size:
            video_embeddings = compute_video_embedding(batch_frames, model, preprocessor, device, dtype)
            for path, video_embedding in zip(batch_paths, video_embeddings):
                if video_embedding is not None:
                    similarity = np.dot(ref_embedding, video_embedding)
                    results.append((path.split("/")[-1], similarity))
            batch_frames = []
            batch_paths = []

    # Remaining.
    if batch_frames:
        video_embeddings = compute_video_embedding(batch_frames, model, preprocessor, device, dtype)
        for path, video_embedding in zip(batch_paths, video_embeddings):
            if video_embedding is not None:
                similarity = np.dot(ref_embedding, video_embedding)
                results.append((path.split("/")[-1], similarity))

    # Sort videos by similarity score (higher means more similar).
    results.sort(key=lambda x: x[1], reverse=True)

    # Write results to a parquet file.
    df = pd.DataFrame(results, columns=["video_path", "similarity"])
    df.to_parquet(args.parquet_out_path, index=False, float_format="%.4f")

    print(f"\nResults saved to {args.parquet_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_folder",
        type=str,
        required=True,
        help="Path to folder containing videos.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference image/video file(s).",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=24,
        help="Max number of frames per videos.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many videos to process.",
    )
    parser.add_argument(
        "--parquet_out_path",
        type=str,
        default="results.parquet",
        help="Path to the output parquet file.",
    )
    args = parser.parse_args()
    main(args)
