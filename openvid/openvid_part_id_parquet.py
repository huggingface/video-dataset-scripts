import pandas as pd
from zipstream import ZipStream
import tqdm

df = pd.read_csv("OpenVid-1M.csv")

part_ids = list(range(0, 183))
for multi_part in {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118}:
    part_ids.remove(multi_part)

url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}.zip?download=true"

filename_part = []

for part_id in tqdm.tqdm(part_ids):
    stream = ZipStream(url.format(part=part_id))
    filename_part.extend(
        [
            {
                "video": file.filename.split("/")[-1],
                "part_id": part_id,
                "file_offset": file.file_offset,
                "file_size": file.file_size,
            }
            for file in stream.files
        ]
    )

# for split parts we get 1 byte of part a to find the size
# for part b the central directory offset is - size of part a
url_multipart_a = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}_partaa?download=true"
url_multipart = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}_partab?download=true"

for part_id in tqdm.tqdm(
    {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118, 183, 184, 185}
):
    offset = ZipStream.size(url_multipart_a.format(part=part_id))
    stream = ZipStream(url_multipart.format(part=part_id), offset=offset)
    filename_part.extend(
        [
            {
                "video": file.filename.split("/")[-1],
                "part_id": part_id,
                "file_offset": file.file_offset,
                "file_size": file.file_size,
            }
            for file in stream.files
        ]
    )

data = pd.DataFrame(filename_part)

df = df.merge(data, how="left")
df["part_id"] = df["part_id"].astype(pd.Int64Dtype())
df["file_offset"] = df["file_offset"].astype(pd.Int64Dtype())
df["file_size"] = df["file_size"].astype(pd.Int64Dtype())

df.to_parquet("openvid.parquet")
