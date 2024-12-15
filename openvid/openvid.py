from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import requests
from tqdm import tqdm
import zlib

PARQUET_PATH = "openvid.parquet"
BASE_PATH = "./openvid"

# skip these for now
MULTI_PART = {73, 76, 78, 83, 88, 89, 92, 95, 96, 102, 103, 111, 118, 183, 184, 185}

URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part}.zip?download=true"


def o(data: bytes, main_offset: int, offset: int, size: int):
    return data[main_offset + offset : main_offset + offset + size]


def le(data: bytes):
    return int.from_bytes(data, byteorder="little")


def get_central_directory(url: str, _offset: int = 0):
    """
    Downloads the last 64kb to find zip64 End of central directory signature
    https://en.wikipedia.org/wiki/ZIP_(file_format)#ZIP64
    TODO: support standard zip End of central directory signature 0x06054b50
    Then finds offset for start of central directory and downloads it
    `_offset` is to support split zip e.g.
        `OpenVid_part{part}_partaa`, `OpenVid_part{part}_partab`
        `_offset` is size of partaa
    """
    tail_size = 65536
    headers = {"Range": f"bytes=-{tail_size}"}
    tail_data = requests.get(url, headers=headers, stream=True).content
    zip64_eocd = b"\x50\x4b\x06\x06"
    eocd_offset = tail_data.rfind(zip64_eocd)
    eocd = tail_data[eocd_offset:]
    cd_offset = int.from_bytes(eocd[48 : 48 + 8], byteorder="little")
    headers = {"Range": f"bytes={cd_offset-_offset}-"}
    central_directory = requests.get(url, headers=headers, stream=True).content
    return central_directory


def get_files(central_directory: bytes, part_id: int, file_to_get: str = None):
    """
    Find all files, offsets and (compressed) file sizes,
    or find `file_to_get`.
    Supports zip and zip64
        the zip file as a whole can be `zip64` but contain `zip` records
    TODO: get compression type
    https://en.wikipedia.org/wiki/ZIP_(file_format)#Central_directory_file_header_(CDFH)
    """
    files = []
    offset = 0
    while offset <= len(central_directory):
        file_name_length = le(o(central_directory, offset, 28, 2))
        extra_length = le(o(central_directory, offset, 30, 2))
        comment_length = le(o(central_directory, offset, 32, 2))
        filename = o(central_directory, offset, 46, file_name_length).decode("utf-8")
        filename = filename.split("/")[-1]
        next_offset = offset + 46 + file_name_length + extra_length + comment_length
        file_size = le(o(central_directory, offset, 20, 4))
        file_offset = int.from_bytes(
            central_directory[offset + 42 : offset + 42 + 4], byteorder="little"
        )
        is_zip64 = (file_size == 2**32 - 1) or (file_offset == 2**32 - 1)
        if is_zip64:
            extra = central_directory[offset + 46 + file_name_length : next_offset]
            file_offset = int.from_bytes(extra[-8:], byteorder="little")
            file_size = int.from_bytes(
                central_directory[offset + 20 : offset + 20 + 4], byteorder="little"
            )
        if file_to_get and file_to_get == filename:
            return filename, file_offset, file_size, part_id
        elif file_to_get is None:
            files.append((filename, file_offset, file_size, part_id))
        offset = next_offset
    return files


def download_file(
    url: str,
    filename: str,
    file_offset: int,
    file_size: int,
    base_dir: str,
    part_id: int,
):
    """
    https://en.wikipedia.org/wiki/ZIP_(file_format)#Local_file_header
    Only supports DEFLATE
    TODO: support other compression types
    TODO: support multipart zip
    """
    if os.path.exists(f"{base_dir}/{filename}"):
        return
    # plus a little extra to account for file_name_lenght and extra_length
    headers = {"Range": f"bytes={file_offset}-{file_offset+file_size+1024}"}
    file_data = requests.get(url, headers=headers, stream=True).content
    if len(file_data) < 1024:
        print(f"smol download {filename} from part {part_id}")
        return
    file_name_length = le(o(file_data, 0, 26, 2))
    extra_length = le(o(file_data, 0, 28, 2))
    data_offset = 30 + file_name_length + extra_length
    compressed_data = o(file_data, 0, data_offset, file_size)
    try:
        data = zlib.decompress(compressed_data, -15)
    except:
        print(f"error downloading {filename} from part {part_id}")
        return
    if len(data) < 1024:
        print(f"smol decompress {filename} from part {part_id}")
        return
    with open(f"{base_dir}/{filename}", "wb") as f:
        f.write(data)


def get_size(url: str):
    """
    Gets the last byte then parses full size from `Content-Range`
    """
    headers = {"Range": f"bytes=-1"}
    return int(
        requests.get(url, headers=headers).headers["Content-Range"].split("/")[-1]
    )


df = pd.read_parquet(PARQUET_PATH)

aesthetic = df.loc[df["aesthetic score"] >= 7]
aesthetic = aesthetic.loc[~df["part_id"].isin(MULTI_PART)]
part_ids = list(aesthetic["part_id"].unique())
filenames = set(aesthetic["video"])

all_files = []

for part_id in tqdm(part_ids, desc="get central directory"):
    cd = get_central_directory(URL.format(part=part_id))
    all_files.extend(get_files(cd, part_id))

all_files = list(filter(lambda all_file: all_file[0] in filenames, all_files))


with ThreadPoolExecutor(max_workers=8) as executor:
    pbar = tqdm(desc="download", total=len(all_files))
    futures = {
        executor.submit(
            download_file,
            URL.format(part=part_id),
            filename,
            file_offset,
            file_size,
            BASE_PATH,
            part_id,
        ): (
            URL.format(part=part_id),
            filename,
            file_offset,
            file_size,
            BASE_PATH,
            part_id,
        )
        for filename, file_offset, file_size, part_id in all_files
    }
    for future in as_completed(futures):
        _ = future.result()
        pbar.update()
