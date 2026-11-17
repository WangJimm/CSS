"""Validate the shipped RP inference bundle against CSS graph identity."""
import hashlib
import json
from pathlib import Path


def validate_route_bundle(path, environment_name, nid2idx):
    path = Path(path)
    manifest = path.parent / "models_manifest.json"
    if not manifest.is_file():
        return  # Legacy/explicit external checkpoints retain their own loaders.
    info = json.loads(manifest.read_text(encoding="utf-8"))
    entry = next((m for m in info["models"] if m["filename"] == path.name), None)
    if entry is None:
        return
    if info["environment"] != environment_name:
        raise ValueError("Route bundle environment differs from selected CSS environment")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Route model differs from verified bundle: {path}")
    actual_nodes = [node for node, _ in sorted(nid2idx.items(), key=lambda pair: pair[1])]
    if actual_nodes != info["node_ids"]:
        raise ValueError("Route model node ordering differs from the CSS graph")
    project = Path(__file__).resolve().parents[3]
    network = project / environment_name / "env_static_tls.net.xml"
    if hashlib.sha256(network.read_bytes()).hexdigest() != info["network_sha256"]:
        raise ValueError("Route model network differs from selected CSS network")
