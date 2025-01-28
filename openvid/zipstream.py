from dataclasses import dataclass
import pathlib
import requests
import struct
import tqdm
from typing import Optional
import zlib


@dataclass
class LocalFileHeader:
    signature: bytes
    version: int
    flag: int
    method: int
    modification_time: int
    modification_date: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    file_name_length: int
    extra_field_length: int


@dataclass
class CentralDirectoryFileHeader:
    signature: bytes
    version: int
    minimum_version: int
    flag: int
    method: int
    modification_time: int
    modification_date: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    file_name_length: int
    extra_field_length: int
    file_comment_length: int
    disk_number: int
    internal_file_attributes: int
    external_file_attributes: int
    relative_offset: int


class ZipStreamFile:
    def __init__(
        self,
        url: str,
        filename: str,
        file_offset: int,
        file_size: int,
    ):
        self.url = url
        self.filename = filename
        self.file_offset = file_offset
        self.file_size = file_size

    def download(
        self,
        filename: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        struct_format = "<4sHHHHHIIIHH"
        struct_size = struct.calcsize(struct_format)
        headers = {"Range": f"bytes={self.file_offset}-{self.file_offset+struct_size-1}"}
        local_file_header = requests.get(self.url, headers=headers, stream=True).content
        local_file_header = LocalFileHeader(*struct.unpack(struct_format, local_file_header))
        data_offset = struct_size + local_file_header.file_name_length + local_file_header.extra_field_length
        headers = {"Range": f"bytes={self.file_offset+data_offset}-{self.file_offset+data_offset+self.file_size-1}"}
        data = requests.get(self.url, headers=headers, stream=True).content
        if local_file_header.method == 8:
            data = zlib.decompress(data, -15)
        elif local_file_header.method != 0:
            raise ValueError("Unsupported compression method.")
        filename = filename or self.filename
        if base_path is not None and filename is not None:
            with open(f"{base_path}/{filename}", "wb") as f:
                f.write(data)
        return data

    def __repr__(self):
        return f"ZipStreamFile(\n\turl={self.url},\n\tfilename={self.filename},\n\tfile_offset={self.file_offset},\n\tfile_size={self.file_size}\n)"


class ZipStream:
    tail_size: int = 65536

    @classmethod
    def size(self, url: str):
        headers = {"Range": f"bytes=-1"}
        return int(requests.get(url, headers=headers).headers["Content-Range"].split("/")[-1])

    @classmethod
    def get_central_directory(self, url: str, offset: Optional[int] = None):
        headers = {"Range": f"bytes=-{self.tail_size}"}
        tail_data = requests.get(url, headers=headers, stream=True).content
        zip64_eocd = b"\x50\x4b\x06\x06"
        eocd_offset = tail_data.rfind(zip64_eocd)
        eocd = tail_data[eocd_offset:]
        cd_offset = int.from_bytes(eocd[48 : 48 + 8], byteorder="little")
        if offset is not None:
            cd_offset - offset
        headers = {"Range": f"bytes={cd_offset}-"}
        central_directory = requests.get(url, headers=headers, stream=True).content
        return central_directory

    @classmethod
    def get_files(self, url: str, central_directory: bytes, file_to_get: str = None):
        files = []
        offset = 0
        while offset <= len(central_directory):
            file, offset = ZipStream.get_file(url=url, central_directory=central_directory, offset=offset)
            if file is None:
                continue
            if file_to_get is None:
                files.append(file)
            elif file_to_get is not None and file_to_get in file.filename:
                return file
        return files

    @classmethod
    def get_file(self, url: str, central_directory: bytes, offset: int):
        struct_format = "<4sHHHHHHIIIHHHHHII"
        struct_size = struct.calcsize(struct_format)
        buffer = central_directory[offset : offset + struct_size]
        if len(buffer) < struct_size:
            return None, offset + struct_size
        central_directory_file_header = CentralDirectoryFileHeader(*struct.unpack(struct_format, buffer))
        filename = central_directory[
            offset + struct_size : offset + struct_size + central_directory_file_header.file_name_length
        ].decode("utf-8")
        next_offset = (
            offset
            + struct_size
            + central_directory_file_header.file_name_length
            + central_directory_file_header.extra_field_length
            + central_directory_file_header.file_comment_length
        )
        if not filename:
            return None, next_offset
        is_zip64 = (central_directory_file_header.compressed_size == 2**32 - 1) or (
            central_directory_file_header.relative_offset == 2**32 - 1
        )
        if is_zip64:
            extra = central_directory[
                offset + struct_size + central_directory_file_header.file_name_length : next_offset
            ]
            central_directory_file_header.relative_offset = int.from_bytes(extra[-8:], byteorder="little")
        return (
            ZipStreamFile(
                url=url,
                filename=filename,
                file_offset=central_directory_file_header.relative_offset,
                file_size=central_directory_file_header.compressed_size,
            ),
            next_offset,
        )

    def __init__(
        self,
        url: str,
        central_directory: Optional[bytes] = None,
        offset: Optional[int] = None,
    ):
        self.url = url
        central_directory = central_directory or ZipStream.get_central_directory(url=self.url, offset=offset)
        self.central_directory = central_directory
        self.files = ZipStream.get_files(url=self.url, central_directory=self.central_directory)
