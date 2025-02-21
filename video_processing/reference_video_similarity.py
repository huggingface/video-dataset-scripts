import os
import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from PIL import Image
import numpy as np
import argparse
import csv
from frames import get_frames


def compute_video_embedding(frames, model, preprocessor, device, batch_size=0):
    """
    Computes an embedding for a video by averaging the embeddings of its frames.
    If batch_size > 0, frames are processed in batches; otherwise, all frames are processed at once.
    """
    if batch_size and batch_size > 0:
        embeddings = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_input = preprocessor(images=batch_frames, return_tensors="pt").to(device)
            with torch.no_grad() and torch.autocast(torch.device(device).type, dtype=torch.float16):
                batch_embeddings = model(**batch_input).pooler_output
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            embeddings.append(batch_embeddings.cpu())
        embeddings = torch.cat(embeddings, dim=0)
    else:
        # Process all frames at once.
        all_input = preprocessor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad() and torch.autocast(torch.device(device).type, dtype=torch.float16):
            embeddings = model(**all_input).pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu()

    video_embedding = embeddings.mean(dim=0)
    video_embedding = video_embedding / video_embedding.norm()
    return video_embedding.numpy()


def compute_image_embedding(image_path, model, preprocess, device):
    """
    Computes an embedding for a single image.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad() and torch.autocast(torch.device(device).type, dtype=torch.float16):
        embedding = model(image_input).pooler_output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def compute_reference_embedding(ref_path, model, preprocessor, device, batch_size=0):
    """
    Computes the embedding for a reference file (image or video).
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    if ref_path.lower().endswith(video_extensions):
        frames = get_frames(ref_path)
        return compute_video_embedding(frames, model, preprocessor, device, batch_size)
    else:
        return compute_image_embedding(ref_path, model, preprocessor, device)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Process each reference file and average their embeddings.
    ref_embeddings = []
    for ref in args.reference:
        emb = compute_reference_embedding(ref, model, preprocessor, device, args.batch_size)
        if emb is not None:
            ref_embeddings.append(emb)
        else:
            print(f"Could not compute embedding for reference: {ref}")

    if len(ref_embeddings) == 0:
        print("No valid reference embeddings found!")
        return

    ref_embedding = np.mean(ref_embeddings, axis=0)
    ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)

    # List video files in the folder (supports common video extensions)
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [
        os.path.join(args.videos_folder, f)
        for f in os.listdir(args.videos_folder)
        if f.lower().endswith(video_extensions)
    ]

    results = []
    for video_path in video_files:
        print(f"Processing {video_path}...")
        frames = get_frames(video_path)
        frames = [frame.to_image() for frame in frames]
        if len(frames) == 0:
            print(f"Could not extract frames from {video_path}")
            continue
        video_embedding = compute_video_embedding(frames, model, preprocessor, device, args.batch_size)
        if video_embedding is None:
            continue
        # Compute cosine similarity between reference and video embeddings.
        similarity = np.dot(ref_embedding, video_embedding)
        results.append((video_path, similarity))

    # Sort videos by similarity score (higher means more similar).
    results.sort(key=lambda x: x[1], reverse=True)

    # Write results to CSV.
    with open(args.output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_path", "similarity"])
        for video, score in results:
            writer.writerow([video, f"{score:.4f}"])

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video similarity inference using CLIP with PyAV")
    parser.add_argument("--videos_folder", type=str, required=True, help="Path to folder containing videos.")
    parser.add_argument("--reference", type=str, nargs="+", required=True, help="Reference image/video file(s).")
    parser.add_argument(
        "--batch_size", type=int, default=0, help="Batch size for inference (default: process all frames at once)."
    )
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Path to the output CSV file.")
    args = parser.parse_args()
    main(args)
