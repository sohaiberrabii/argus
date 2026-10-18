"""Download a pinned, real BF16 Llama checkpoint on the TPU VM.

Uses the public Unsloth mirror, no token, random weights or quantization.
The upstream model loader consumes the resulting local directory.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "unsloth/Llama-3.2-1B-Instruct"
REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
WEIGHTS_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"


def main():
    dest = Path(sys.argv[1]).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"))
    assert info["sha"] == REVISION
    record = {"repository": REPO, "revision": REVISION, "base_model": info["cardData"]["base_model"],
              "parameters": info["safetensors"], "files": []}
    for item in info["siblings"]:
        name = item["rfilename"]
        if not (name.endswith((".json", ".safetensors", ".jinja"))):
            continue
        path = dest / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            tmp = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", tmp)
            tmp.rename(path)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if name == "model.safetensors" and digest != WEIGHTS_SHA256:
            raise ValueError("Pinned BF16 checkpoint hash mismatch")
        record["files"].append({"name": name, "bytes": path.stat().st_size, "sha256": digest})
    (dest / "checkpoint-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
