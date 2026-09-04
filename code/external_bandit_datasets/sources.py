"""Pinned source records and checksum-verifying downloads."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    dataset_id: str
    species: str
    title: str
    repository: str
    doi: str
    version: str
    license: str
    url: str
    filename: str
    digest_algorithm: str
    digest: str
    archive_member: str | None = None


SOURCES: dict[str, Source] = {
    "grossman": Source(
        dataset_id="grossman-bari-cohen-2021",
        species="mouse",
        title="Serotonin neurons modulate learning rate through uncertainty",
        repository="Dryad",
        doi="10.5061/dryad.cz8w9gj4s",
        version="4 (2021-12-27; Dryad resource 156295)",
        license="CC0-1.0",
        url="https://datadryad.org/api/v2/versions/156295/download",
        filename="grossmanBariCohenData.zip",
        digest_algorithm="sha256",
        digest="43a19b171f88430d524557a5c2e13518d6d37ffcfa1dcddc4159b420b1f0485a",
        archive_member="grossmanBariCohenData.zip",
    ),
    "chen": Source(
        dataset_id="chen-et-al-2021",
        species="mouse",
        title="Sex differences in learning from exploration",
        repository="Dryad",
        doi="10.5061/dryad.z612jm6c0",
        version="5 (2022-02-07; Dryad resource 162666)",
        license="CC0-1.0",
        url="https://datadryad.org/api/v2/versions/162666/download",
        filename="cleaned_up_restless_final_data.zip",
        digest_algorithm="sha256",
        digest="90f0f9fa843a16788d0dcd7b857f81db068e8d18b8dd4eabf20ccaee3b67db04",
        archive_member="cleaned_up_restless_final_data.zip",
    ),
    "zid": Source(
        dataset_id="zid-et-al-2026-experiment-1",
        species="human",
        title="Foraging models explain human exploration in uncertain tasks",
        repository="Figshare",
        doi="10.6084/m9.figshare.32193990.v5",
        version="5 (Figshare file 64311972)",
        license="MIT",
        url="https://ndownloader.figshare.com/files/64311972",
        filename="all_sub_2ab.pickle",
        digest_algorithm="md5",
        digest="bfdfcc37d1e1a0aa66f31ba99ca89140",
    ),
}


def file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_source_file(path: Path, source: Source) -> None:
    actual = file_digest(path, source.digest_algorithm)
    if actual != source.digest:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {source.digest}, got {actual}."
        )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/zip, application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)


def download_source(name: str, raw_root: str | Path, *, force: bool = False) -> Path:
    """Download one pinned source and return the checksum-verified payload path."""
    source = SOURCES[name]
    source_dir = Path(raw_root) / name
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / source.filename
    if destination.exists() and not force:
        verify_source_file(destination, source)
        return destination

    temporary = source_dir / f".{source.filename}.download"
    if temporary.exists():
        temporary.unlink()
    _download(source.url, temporary)
    if source.archive_member is not None:
        with zipfile.ZipFile(temporary) as archive:
            member = archive.getinfo(source.archive_member)
            if Path(member.filename).name != member.filename:
                raise ValueError(f"Unsafe archive member name: {member.filename!r}.")
            with archive.open(member) as input_stream, destination.open("wb") as output:
                shutil.copyfileobj(input_stream, output)
        temporary.unlink()
    else:
        temporary.replace(destination)
    verify_source_file(destination, source)
    return destination
