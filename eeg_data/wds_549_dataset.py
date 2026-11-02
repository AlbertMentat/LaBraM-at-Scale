"""Strict iterator for WDS 549 full-release tar triples."""
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


class WDS549StrictReader:
    def __init__(self, root, catalog_path=None, verify_catalog=True):
        self.root = Path(root)
        self.catalog_path = Path(catalog_path or self.root / "build_catalog.json")
        self.catalog = json.loads(self.catalog_path.read_text())
        if verify_catalog:
            from tools.wds_549_full_io import scan_catalog
            scan_catalog(self.root, self.catalog_path, verify_hashes=True)

    def __len__(self):
        return int(self.catalog["windows"])

    def __iter__(self):
        for record in self.catalog["tar_files"]:
            with tarfile.open(self.root / record["path"]) as archive:
                samples = defaultdict(dict)
                for member in archive.getmembers():
                    if member.name.endswith(".mask.npy"):
                        samples[member.name[:-9]]["mask"] = member
                    elif member.name.endswith(".npy"):
                        samples[member.name[:-4]]["data"] = member
                    elif member.name.endswith(".json"):
                        samples[member.name[:-5]]["metadata"] = member
                for key in sorted(samples):
                    parts = samples[key]
                    data = np.load(
                        io.BytesIO(archive.extractfile(parts["data"]).read()),
                        allow_pickle=False)
                    mask = np.load(
                        io.BytesIO(archive.extractfile(parts["mask"]).read()),
                        allow_pickle=False)
                    metadata = json.loads(
                        archive.extractfile(parts["metadata"]).read())
                    yield {"eeg": data, "padding_mask": mask, "metadata": metadata}
