"""Read remote ZIP archives with HTTP range requests."""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import requests

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"

# Bound coalesced reads to keep memory stable.
MAX_SPAN_BYTES = 16 << 20
# Do not bridge large gaps between ZIP members.
MAX_GAP_BYTES = 1 << 20

MAX_RETRIES = 4
RETRY_BACKOFF = 1.7

# Static archives permit modest parallel range requests.
DEFAULT_WORKERS = 8


class RemoteZipError(Exception):
    pass


@dataclass(frozen=True)
class Member:
    """One entry from the central directory."""

    name: str
    compress_type: int
    compress_size: int
    file_size: int
    header_offset: int
    crc: int
    name_len: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")

    def max_local_span(self) -> int:
        """Return a safe upper bound for a member range request."""
        return 30 + self.name_len + 4096 + self.compress_size


class RemoteZip:
    """Random access to a ZIP over HTTP, without downloading the archive."""

    def __init__(
        self,
        url: str,
        size: int | None = None,
        *,
        session: requests.Session | None = None,
        cache_dir: str | Path | None = "cache/zip_index",
        timeout: int = 120,
    ):
        self.url = url
        self.timeout = timeout
        self._shared_session = session
        self._local = threading.local()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.size = size if size is not None else self._probe_size()
        self._members: list[Member] | None = None

    @property
    def session(self) -> requests.Session:
        """Return one HTTP session per worker thread."""
        if self._shared_session is not None:
            return self._shared_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _probe_size(self) -> int:
        response = self.session.head(
            self.url, allow_redirects=True, timeout=self.timeout
        )
        response.raise_for_status()
        if "Content-Length" not in response.headers:
            raise RemoteZipError(f"server did not report a size for {self.url}")
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RemoteZipError(
                f"server does not advertise byte ranges for {self.url}; "
                "whole-archive download is the only option"
            )
        return int(response.headers["Content-Length"])

    def _range(self, start: int, end: int) -> bytes:
        """Fetch an inclusive byte range with retries."""
        start = max(0, start)
        end = min(end, self.size - 1)
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=self.timeout,
                )
                if response.status_code not in (200, 206):
                    raise RemoteZipError(
                        f"expected 206 for a ranged request, got {response.status_code}"
                    )
                data = response.content
                if response.status_code == 200 and len(data) > (end - start + 1):
                    # Some servers ignore Range and return the full file.
                    data = data[start : end + 1]
                return data
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF**attempt)

        raise RemoteZipError(
            f"range {start}-{end} failed after {MAX_RETRIES} tries: {last_error}"
        )

    def _cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(f"{self.url}|{self.size}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}.json"

    def members(self) -> list[Member]:
        """Every entry, read once and cached to disk."""
        if self._members is not None:
            return self._members

        cache = self._cache_path()
        if cache and cache.exists():
            raw = json.loads(cache.read_text())
            self._members = [Member(**m) for m in raw]
            return self._members

        self._members = self._read_central_directory()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([m.__dict__ for m in self._members]))
        return self._members

    def _read_central_directory(self) -> list[Member]:
        tail_len = min(65_536 + 22, self.size)
        tail = self._range(self.size - tail_len, self.size - 1)

        eocd_at = tail.rfind(EOCD_SIG)
        if eocd_at == -1:
            raise RemoteZipError("no end-of-central-directory record; not a ZIP?")
        _, cd_size, cd_offset = struct.unpack("<HII", tail[eocd_at + 10 : eocd_at + 20])

        # ZIP64 stores real sizes outside fields set to 0xffffffff.
        if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            locator_at = tail.rfind(EOCD64_LOCATOR_SIG)
            if locator_at == -1:
                raise RemoteZipError("zip64 sizes indicated but no zip64 locator found")
            eocd64_offset = struct.unpack("<Q", tail[locator_at + 8 : locator_at + 16])[
                0
            ]
            eocd64 = self._range(eocd64_offset, eocd64_offset + 55)
            if eocd64[:4] != EOCD64_SIG:
                raise RemoteZipError(
                    "zip64 end-of-central-directory signature mismatch"
                )
            cd_size, cd_offset = struct.unpack("<QQ", eocd64[40:56])

        return self._parse_central_directory(
            self._range(cd_offset, cd_offset + cd_size - 1)
        )

    @staticmethod
    def _parse_central_directory(cd: bytes) -> list[Member]:
        members: list[Member] = []
        pos = 0
        while pos + 46 <= len(cd) and cd[pos : pos + 4] == CENTRAL_SIG:
            method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
            crc, compress_size, file_size = struct.unpack(
                "<III", cd[pos + 16 : pos + 28]
            )
            name_len, extra_len, comment_len = struct.unpack(
                "<HHH", cd[pos + 28 : pos + 34]
            )
            header_offset = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
            name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
            extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]

            if 0xFFFFFFFF in (compress_size, file_size, header_offset):
                file_size, compress_size, header_offset = _apply_zip64_extra(
                    extra, file_size, compress_size, header_offset
                )

            members.append(
                Member(
                    name=name,
                    compress_type=method,
                    compress_size=compress_size,
                    file_size=file_size,
                    header_offset=header_offset,
                    crc=crc,
                    name_len=name_len,
                )
            )
            pos += 46 + name_len + extra_len + comment_len

        if not members:
            raise RemoteZipError("central directory parsed to zero entries")
        return members

    def namelist(self) -> list[str]:
        return [m.name for m in self.members()]

    @staticmethod
    def _extract_from_span(member: Member, span: bytes, span_start: int) -> bytes:
        local = member.header_offset - span_start
        if span[local : local + 4] != LOCAL_SIG:
            raise RemoteZipError(f"{member.name}: local header signature mismatch")
        name_len, extra_len = struct.unpack("<HH", span[local + 26 : local + 30])
        data_at = local + 30 + name_len + extra_len
        raw = span[data_at : data_at + member.compress_size]

        if len(raw) < member.compress_size:
            raise RemoteZipError(
                f"{member.name}: short read ({len(raw)} of {member.compress_size})"
            )

        if member.compress_type == 0:
            data = raw
        elif member.compress_type == 8:
            data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
        else:
            raise RemoteZipError(
                f"{member.name}: unsupported compression method {member.compress_type}"
            )

        if member.crc and zlib.crc32(data) & 0xFFFFFFFF != member.crc:
            raise RemoteZipError(f"{member.name}: CRC mismatch")
        return data

    def read(self, member: Member) -> bytes:
        span_end = member.header_offset + member.max_local_span()
        span = self._range(member.header_offset, span_end)
        return self._extract_from_span(member, span, member.header_offset)

    def _read_group(
        self, group: list[Member]
    ) -> list[tuple[Member, bytes | None, str | None]]:
        start = group[0].header_offset
        end = group[-1].header_offset + group[-1].max_local_span()
        out: list[tuple[Member, bytes | None, str | None]] = []
        try:
            span = self._range(start, end)
        except Exception:  # noqa: BLE001 - fall back to one request per member
            for member in group:
                try:
                    out.append((member, self.read(member), None))
                except Exception as inner:  # noqa: BLE001
                    out.append((member, None, f"{inner.__class__.__name__}: {inner}"))
            return out

        for member in group:
            try:
                out.append((member, self._extract_from_span(member, span, start), None))
            except Exception as exc:  # noqa: BLE001 - one bad member, not the batch
                out.append((member, None, f"{exc.__class__.__name__}: {exc}"))
        return out

    def read_many(
        self,
        members: Sequence[Member],
        *,
        max_span: int = MAX_SPAN_BYTES,
        workers: int = DEFAULT_WORKERS,
    ) -> Iterator[tuple[Member, bytes | None, str | None]]:
        """Yield member data while coalescing nearby range requests."""
        ordered = sorted(members, key=lambda m: m.header_offset)
        groups = list(_coalesce(ordered, max_span=max_span))

        if workers <= 1:
            for group in groups:
                yield from self._read_group(group)
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(self._read_group, groups):
                yield from result


def _apply_zip64_extra(
    extra: bytes, file_size: int, compress_size: int, header_offset: int
) -> tuple[int, int, int]:
    """Read conditional 64-bit values from a ZIP64 extra field."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + size]
        if tag == 0x0001:
            cursor = 0
            if file_size == 0xFFFFFFFF and cursor + 8 <= len(body):
                file_size = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if compress_size == 0xFFFFFFFF and cursor + 8 <= len(body):
                compress_size = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if header_offset == 0xFFFFFFFF and cursor + 8 <= len(body):
                header_offset = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            break
        pos += 4 + size
    return file_size, compress_size, header_offset


def _coalesce(
    ordered: Sequence[Member], *, max_span: int, max_gap: int = MAX_GAP_BYTES
) -> Iterator[list[Member]]:
    """Group offset-sorted members into batches that share one range request."""
    group: list[Member] = []
    group_start = 0
    previous_end = 0

    for member in ordered:
        member_end = member.header_offset + member.max_local_span()
        if not group:
            group, group_start, previous_end = (
                [member],
                member.header_offset,
                member_end,
            )
            continue

        gap = member.header_offset - previous_end
        if member_end - group_start > max_span or gap > max_gap:
            yield group
            group, group_start, previous_end = (
                [member],
                member.header_offset,
                member_end,
            )
        else:
            group.append(member)
            previous_end = max(previous_end, member_end)

    if group:
        yield group


def modelscope_url(repo_id: str, path: str, revision: str = "master") -> str:
    """The file endpoint for a ModelScope dataset blob."""
    return (
        f"https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo"
        f"?Revision={revision}&FilePath={path}"
    )
