"""Prepare the pinned BGE-M3 model inside .venv; CUDA is not required."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".venv/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
PINNED_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cpu.json"))
    parser.add_argument("--offline", action="store_true",
                        help="Check already downloaded files without network access")
    args = parser.parse_args(argv)
    from task4_rank_evidence import validated_config, _get_semantic_model
    config_path = Path(args.config).resolve()
    overrides = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = validated_config(overrides)["semantic_model"]
    revision = cfg.get("revision") or PINNED_REVISION
    local = ROOT / ".venv/models/bge-m3"
    required = ("config.json", "model.safetensors", "modules.json",
                "tokenizer.json", "1_Pooling/config.json")
    missing = [name for name in required if not (local / name).is_file()]
    if missing:
        if args.offline:
            raise SystemExit("Local model is incomplete. Run without --offline to download it. Missing: "
                             + ", ".join(missing))
        from huggingface_hub import snapshot_download
        snapshot_download(
            "BAAI/bge-m3", revision=revision, local_dir=str(local),
            allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "1_Pooling/*"],
        )
    cfg.update(name=str(local), revision=revision, local_files_only=True,
               cache_folder=str(ROOT / ".venv/models"))
    model = _get_semantic_model(cfg)
    embedding = model.encode(["Model readiness check."], normalize_embeddings=True,
                             convert_to_numpy=True)
    manifest_path = ROOT / "artifacts/model_manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with (local / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest.update(
        name="BAAI/bge-m3", revision=revision, local_path=".venv/models/bge-m3",
        device=str(model.device), embedding_shape=list(embedding.shape),
        weights_sha256=digest, weights_bytes=(local / "model.safetensors").stat().st_size,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    overrides.setdefault("semantic_model", {}).update(
        name=".venv/models/bge-m3", revision=revision, local_files_only=True,
        cache_folder=".venv/models")
    config_path.write_text(json.dumps(overrides, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    main()
