"""Concurrent, bounded, node-local cache for WDS tar files."""

import errno
import fcntl
import hashlib
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


GIB = 1024 ** 3
DEFAULT_MIN_FREE_BYTES = 20 * GIB
AUTO_CACHE_FRACTION = 0.8
RECONCILE_INTERVAL_SECONDS = 300
STALE_RESERVATION_SECONDS = 3600


def _valid_sha256(value):
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))


class LocalTarCache:
    """Local-first tar cache with flock deduplication and bounded LRU eviction."""

    def __init__(
            self, root, max_bytes=None,
            min_free_bytes=DEFAULT_MIN_FREE_BYTES, verbose=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_root = self.root / ".locks"
        self.lock_root.mkdir(exist_ok=True)
        self.database_path = self.root / ".wds-cache.sqlite3"
        self.global_lock_path = self.root / ".cache-global.lock"
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.verbose = bool(verbose)
        self._initialize_index()
        existing_bytes = self.stats()["bytes"]
        if max_bytes is None or int(max_bytes) <= 0:
            free_bytes = shutil.disk_usage(self.root).free
            usable_capacity = max(
                1, free_bytes + existing_bytes - self.min_free_bytes)
            self.max_bytes = max(
                1, int(usable_capacity * AUTO_CACHE_FRACTION))
        else:
            self.max_bytes = int(max_bytes)
        if self.verbose:
            print(
                "LocalTarCache: root=%s budget=%.1f GiB reserve=%.1f GiB"
                % (self.root, self.max_bytes / GIB,
                   self.min_free_bytes / GIB),
                flush=True)

    def _connect(self):
        return sqlite3.connect(self.database_path, timeout=60)

    @contextmanager
    def _global_lock(self):
        with open(self.global_lock_path, "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _entry_lock_path(self, relative):
        digest = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()
        return self.lock_root / f"{digest}.lock"

    def _initialize_index(self):
        with self._global_lock():
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS entries (
                        path TEXT PRIMARY KEY,
                        size_bytes INTEGER NOT NULL,
                        sha256 TEXT NOT NULL,
                        atime_ns INTEGER NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS reservations (
                        path TEXT PRIMARY KEY,
                        size_bytes INTEGER NOT NULL,
                        created_ns INTEGER NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS counters (
                        name TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        name TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)
                now_ns = time.time_ns()
                stale_before = now_ns - STALE_RESERVATION_SECONDS * 10 ** 9
                connection.execute(
                    "DELETE FROM reservations WHERE created_ns < ?",
                    (stale_before,))
                row = connection.execute(
                    "SELECT value FROM metadata WHERE name='last_reconcile_ns'"
                ).fetchone()
                last_reconcile = int(row[0]) if row else 0
                if now_ns - last_reconcile < RECONCILE_INTERVAL_SECONDS * 10 ** 9:
                    return
                actual = {}
                for path in self.root.rglob("*.tar"):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(self.root).as_posix()
                    sidecar = Path(str(path) + ".sha256")
                    digest = sidecar.read_text().strip() if sidecar.is_file() else ""
                    if not _valid_sha256(digest):
                        lock_path = self._entry_lock_path(relative)
                        with open(lock_path, "a+b") as entry_lock:
                            try:
                                fcntl.flock(
                                    entry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            # Recheck after locking in case a writer published the
                            # sidecar between the first read and lock acquisition.
                            digest = (
                                sidecar.read_text().strip()
                                if sidecar.is_file() else "")
                            if not _valid_sha256(digest):
                                path.unlink(missing_ok=True)
                                sidecar.unlink(missing_ok=True)
                                connection.execute("""
                                    INSERT INTO counters(name, value)
                                    VALUES ('orphan_cleanups', 1)
                                    ON CONFLICT(name) DO UPDATE SET value=value + 1
                                """)
                                fcntl.flock(entry_lock, fcntl.LOCK_UN)
                                continue
                            fcntl.flock(entry_lock, fcntl.LOCK_UN)
                    actual[relative] = (
                        path.stat().st_size, digest, path.stat().st_mtime_ns)
                indexed = {
                    row[0] for row in connection.execute("SELECT path FROM entries")}
                for relative, (size_bytes, digest, atime_ns) in actual.items():
                    connection.execute("""
                        INSERT INTO entries(path, size_bytes, sha256, atime_ns)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            size_bytes=excluded.size_bytes,
                            sha256=excluded.sha256,
                            atime_ns=excluded.atime_ns
                    """, (relative, size_bytes, digest, atime_ns))
                for relative in indexed - set(actual):
                    connection.execute(
                        "DELETE FROM entries WHERE path=?", (relative,))
                connection.execute("""
                    INSERT INTO metadata(name, value)
                    VALUES ('last_reconcile_ns', ?)
                    ON CONFLICT(name) DO UPDATE SET value=excluded.value
                """, (str(now_ns),))

    def _increment_many(self, values):
        with self._connect() as connection:
            connection.executemany("""
                INSERT INTO counters(name, value) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET value=value + excluded.value
            """, [(name, int(value)) for name, value in values.items()])

    def stats(self):
        with self._connect() as connection:
            entries, size_bytes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM entries"
            ).fetchone()
            reservations = connection.execute(
                "SELECT COUNT(*) FROM reservations").fetchone()[0]
            counters = dict(connection.execute(
                "SELECT name, value FROM counters"))
        return {
            "entries": int(entries),
            "bytes": int(size_bytes),
            "reservations": int(reservations),
            "hits": int(counters.get("hits", 0)),
            "hit_bytes": int(counters.get("hit_bytes", 0)),
            "downloads": int(counters.get("downloads", 0)),
            "download_bytes": int(counters.get("download_bytes", 0)),
            "fallbacks": int(counters.get("fallbacks", 0)),
            "fallback_bytes": int(counters.get("fallback_bytes", 0)),
            "evictions": int(counters.get("evictions", 0)),
            "evicted_bytes": int(counters.get("evicted_bytes", 0)),
            "orphan_cleanups": int(counters.get("orphan_cleanups", 0)),
        }

    def summary(self):
        values = self.stats()
        return (
            "entries=%d size=%.1fGiB hits=%d hit=%.1fGiB "
            "downloads=%d blob=%.1fGiB fallbacks=%d fallback=%.1fGiB "
            "evictions=%d evicted=%.1fGiB"
            % (values["entries"], values["bytes"] / GIB,
               values["hits"], values["hit_bytes"] / GIB,
               values["downloads"], values["download_bytes"] / GIB,
               values["fallbacks"], values["fallback_bytes"] / GIB,
               values["evictions"], values["evicted_bytes"] / GIB))

    def _cache_paths(self, relative):
        cached = self.root / relative
        return cached, Path(str(cached) + ".sha256")

    def _is_valid(self, cached, sidecar, expected_size, expected_sha256):
        if not cached.is_file() or cached.stat().st_size != expected_size:
            return False
        if not sidecar.is_file():
            return False
        actual_sha256 = sidecar.read_text().strip()
        if not _valid_sha256(actual_sha256):
            return False
        return expected_sha256 is None or actual_sha256 == expected_sha256

    def _touch(self, relative, cached, digest):
        now_ns = time.time_ns()
        os.utime(cached, ns=(now_ns, now_ns))
        size_bytes = cached.stat().st_size
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO entries(path, size_bytes, sha256, atime_ns)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256,
                    atime_ns=excluded.atime_ns
            """, (relative, size_bytes, digest, now_ns))
            connection.executemany("""
                INSERT INTO counters(name, value) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET value=value + excluded.value
            """, (("hits", 1), ("hit_bytes", size_bytes)))

    def _evict_locked(self, connection, required_bytes, exclude):
        if required_bytes <= 0:
            return 0
        reserve = min(self.max_bytes // 20, 20 * GIB)
        target = required_bytes + reserve
        freed = 0
        candidates = connection.execute(
            "SELECT path, size_bytes FROM entries ORDER BY atime_ns, path"
        ).fetchall()
        for relative, indexed_size in candidates:
            if relative == exclude:
                continue
            lock_path = self._entry_lock_path(relative)
            with open(lock_path, "a+b") as entry_lock:
                try:
                    fcntl.flock(entry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                cached, sidecar = self._cache_paths(relative)
                actual_size = cached.stat().st_size if cached.is_file() else 0
                cached.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                connection.execute(
                    "DELETE FROM entries WHERE path=?", (relative,))
                freed += actual_size
                connection.execute("""
                    INSERT INTO counters(name, value) VALUES ('evictions', 1)
                    ON CONFLICT(name) DO UPDATE SET value=value + 1
                """)
                connection.execute("""
                    INSERT INTO counters(name, value) VALUES ('evicted_bytes', ?)
                    ON CONFLICT(name) DO UPDATE SET value=value + excluded.value
                """, (actual_size,))
                fcntl.flock(entry_lock, fcntl.LOCK_UN)
            if freed >= target:
                break
        return freed

    def _reserve(self, relative, size_bytes):
        with self._global_lock():
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cached_bytes = connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
                ).fetchone()[0]
                reserved_bytes = connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM reservations"
                ).fetchone()[0]
                free_bytes = shutil.disk_usage(self.root).free
                required_budget = max(
                    0, cached_bytes + reserved_bytes + size_bytes - self.max_bytes)
                required_disk = max(
                    0, self.min_free_bytes + reserved_bytes + size_bytes - free_bytes)
                required = max(required_budget, required_disk)
                self._evict_locked(connection, required, exclude=relative)
                cached_after = connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
                ).fetchone()[0]
                free_after = shutil.disk_usage(self.root).free
                fits_budget = (
                    cached_after + reserved_bytes + size_bytes <= self.max_bytes)
                fits_disk = (
                    free_after >= self.min_free_bytes + reserved_bytes + size_bytes)
                if fits_budget and fits_disk:
                    connection.execute("""
                        INSERT OR REPLACE INTO reservations(path, size_bytes, created_ns)
                        VALUES (?, ?, ?)
                    """, (relative, size_bytes, time.time_ns()))
                    connection.commit()
                    return
                # File deletions cannot be rolled back; retain the corrected LRU
                # index even when this request must fall back to the source path.
                connection.commit()
                raise OSError(
                    errno.ENOSPC,
                    "local WDS cache cannot satisfy capacity reserve")

    def _release_reservation(self, relative):
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM reservations WHERE path=?", (relative,))

    def _commit_download(self, relative, cached, digest):
        now_ns = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM reservations WHERE path=?", (relative,))
            connection.execute("""
                INSERT INTO entries(path, size_bytes, sha256, atime_ns)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    sha256=excluded.sha256,
                    atime_ns=excluded.atime_ns
            """, (relative, cached.stat().st_size, digest, now_ns))
            connection.execute("""
                INSERT INTO counters(name, value) VALUES ('downloads', 1)
                ON CONFLICT(name) DO UPDATE SET value=value + 1
            """)
            connection.execute("""
                INSERT INTO counters(name, value) VALUES ('download_bytes', ?)
                ON CONFLICT(name) DO UPDATE SET value=value + excluded.value
            """, (cached.stat().st_size,))
            connection.commit()

    @contextmanager
    def open(
            self, source_path, relative_path, expected_size,
            expected_sha256=None):
        source_path = Path(source_path)
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe local cache path: {relative_path}")
        relative_text = relative.as_posix()
        expected_size = int(expected_size)
        if expected_size <= 0:
            raise ValueError("expected cache size must be positive")
        if expected_sha256 is not None and not _valid_sha256(expected_sha256):
            raise ValueError("expected cache SHA-256 is invalid")
        cached, sidecar = self._cache_paths(relative)
        lock_path = self._entry_lock_path(relative_text)
        with open(lock_path, "a+b") as entry_lock:
            fcntl.flock(entry_lock, fcntl.LOCK_SH)
            if self._is_valid(
                    cached, sidecar, expected_size, expected_sha256):
                digest = sidecar.read_text().strip()
                self._touch(relative_text, cached, digest)
                try:
                    yield cached
                finally:
                    fcntl.flock(entry_lock, fcntl.LOCK_UN)
                return
            fcntl.flock(entry_lock, fcntl.LOCK_UN)
            fcntl.flock(entry_lock, fcntl.LOCK_EX)
            if self._is_valid(
                    cached, sidecar, expected_size, expected_sha256):
                digest = sidecar.read_text().strip()
                self._touch(relative_text, cached, digest)
                try:
                    yield cached
                finally:
                    fcntl.flock(entry_lock, fcntl.LOCK_UN)
                return

            cached.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            temporary = Path(str(cached) + f".tmp.{os.getpid()}")
            sidecar_temporary = Path(str(sidecar) + f".tmp.{os.getpid()}")
            fallback = False
            reserved = False
            try:
                self._reserve(relative_text, expected_size)
                reserved = True
                cached.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with open(source_path, "rb") as source, open(temporary, "wb") as target:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                actual_sha256 = digest.hexdigest()
                if temporary.stat().st_size != expected_size:
                    raise ValueError(f"cached tar size mismatch: {source_path}")
                if (expected_sha256 is not None
                        and actual_sha256 != expected_sha256):
                    raise ValueError(f"cached tar SHA mismatch: {source_path}")
                os.replace(temporary, cached)
                sidecar_temporary.write_text(actual_sha256 + "\n")
                os.replace(sidecar_temporary, sidecar)
                self._commit_download(relative_text, cached, actual_sha256)
                reserved = False
            except (OSError, sqlite3.Error) as error:
                fallback = True
                cached.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                if self.verbose:
                    print(
                        f"LocalTarCache: source fallback for {source_path}: "
                        f"{type(error).__name__}: {error}", flush=True)
                self._increment_many({
                    "fallbacks": 1, "fallback_bytes": expected_size})
            finally:
                if reserved:
                    self._release_reservation(relative_text)
                temporary.unlink(missing_ok=True)
                sidecar_temporary.unlink(missing_ok=True)

            try:
                yield source_path if fallback else cached
            finally:
                fcntl.flock(entry_lock, fcntl.LOCK_UN)
