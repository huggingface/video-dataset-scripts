import requests

import pandas as pd

df = pd.read_csv("OpenVid-1M.csv")


def get_filenames(url: str, _offset=0):
    tail_size = 65536
    headers = {"Range": f"bytes=-{tail_size}"}
    tail_data = requests.get(url, headers=headers, stream=True).content
    zip64_eocd = b"\x50\x4b\x06\x06"
    eocd_offset = tail_data.rfind(zip64_eocd)
    eocd = tail_data[eocd_offset:]
    cd_offset = int.from_bytes(eocd[48 : 48 + 8], byteorder="little")
    headers = {"Range": f"bytes={cd_offset-_offset}-"}
    central_directory = requests.get(url, headers=headers, stream=True).content
    filenames = []
    offset = 0
    while offset <= len(central_directory):
        file_name_length = int.from_bytes(
            central_directory[offset + 28 : offset + 28 + 2], byteorder="little"
        )
        extra_length = int.from_bytes(
            central_directory[offset + 30 : offset + 30 + 2], byteorder="little"
        )
        comment_length = int.from_bytes(
            central_directory[offset + 32 : offset + 32 + 2], byteorder="little"
        )
        filename = central_directory[
            offset + 46 : offset + 46 + file_name_length
        ].decode("utf-8")
        filename = filename.split("/")[-1]
        if filename:
            filenames.append(filename)
        offset += 46 + file_name_length + extra_length + comment_length
    return filenames


part_ids = list(range(0, 183))
for multi_part in {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118}:
    part_ids.remove(multi_part)

url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}.zip?download=true"

filename_part = []

for part_id in part_ids:
    filenames = get_filenames(url=url.format(part=part_id))
    filename_part.extend(
        [{"video": filename, "part_id": part_id} for filename in filenames]
    )

# for split parts we get 1 byte of part a to find the size
# for part b the central directory offset is - size of part a
headers = {"Range": f"bytes=-1"}
url_multipart_a = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}_partaa?download=true"
url_multipart = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}_partab?download=true"

for part_id in {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118, 183, 184, 185}:
    first_part_size = int(
        requests.get(url_multipart_a.format(part=part_id), headers=headers)
        .headers["Content-Range"]
        .split("/")[-1]
    )
    filenames = get_filenames(
        url=url_multipart.format(part=part_id), _offset=first_part_size
    )
    filename_part.extend(
        [{"video": filename, "part_id": part_id} for filename in filenames]
    )

data = pd.DataFrame(filename_part)

df = df.merge(data, how="left")
df["part_id"] = df["part_id"].astype(pd.Int64Dtype())

df.to_parquet("openvid.parquet")
