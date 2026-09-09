"""Build a reviewable public-source ZIP; exclude inputs, outputs, weights and tokens."""
import argparse
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "inputs", "outputs", "cache", "dist", "__pycache__", ".pytest_cache",
            ".ipynb_checkpoints", ".venv", "venv", "env", "tmp", ".idea", ".vscode", "node_modules"}
SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors", ".npz", ".blend"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/Pixal3D-vast-bumper-source.zip")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
            continue
        if path == output or path.suffix in SUFFIXES or path.name.startswith(".env") or path.name == "cache-before-install.txt":
            continue
        if path.name.endswith(".review.json"):
            continue
        data = path.read_bytes()
        if path.suffix in (".py", ".md", ".json", ".sh", ".txt", ".ipynb"):
            if re.search(rb"hf_[A-Za-z0-9]{20,}", data):
                raise ValueError(f"Possible HF credential in {relative}; package not created")
        if path.suffix == ".ipynb":
            notebook = json.loads(data)
            notebook.get("metadata", {}).pop("widgets", None)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            data = json.dumps(notebook, ensure_ascii=False, indent=1).encode("utf-8")
        files.append((relative.as_posix(), data))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr("Pixal3D/"+name, data)
    print(f"Packaged {len(files)} files: {output}")


if __name__ == "__main__":
    main()
