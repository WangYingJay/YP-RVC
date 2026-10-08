#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import pathlib
import shutil
import site

src = pathlib.Path("tools/rvc_py311_compat.py")
site_dir = pathlib.Path(site.getsitepackages()[0])
shutil.copy(src, site_dir / "rvc_py311_compat.py")
(site_dir / "rvc_py311_compat.pth").write_text("import rvc_py311_compat\n", encoding="utf-8")
print("installed collections compat into", site_dir)
PY

python -m pip install --no-cache-dir "cython" "torchaudio==2.4.1" "numpy>=1.23,<2"

work=/tmp/fairseq-build
rm -rf "$work"
mkdir -p "$work"
curl -fL --retry 3 -o "$work/fairseq-0.12.2.tar.gz" \
  "https://pypi.tuna.tsinghua.edu.cn/packages/30/36/db42846570f479ac859be3b48d8b47f2ae9b0b9c77487a512f2f2ecbcb6b/fairseq-0.12.2.tar.gz" \
  || curl -fL --retry 3 -o "$work/fairseq-0.12.2.tar.gz" \
  "https://files.pythonhosted.org/packages/30/36/db42846570f479ac859be3b48d8b47f2ae9b0b9c77487a512f2f2ecbcb6b/fairseq-0.12.2.tar.gz"
tar -xzf "$work/fairseq-0.12.2.tar.gz" -C "$work"

python - <<'PY'
import os
from pathlib import Path

setup_py = Path("/tmp/fairseq-build/fairseq-0.12.2/setup.py")
text = setup_py.read_text(encoding="utf-8")
needle = "def do_setup(package_data):\n"
insert = """def do_setup(package_data):
    global extensions
    kept = []
    for ext in extensions:
        missing = [src for src in ext.sources if not os.path.exists(src)]
        if missing:
            print("skip fairseq extension", ext.name, missing)
            continue
        kept.append(ext)
    extensions = kept
"""
if needle not in text:
    raise SystemExit("无法修改 fairseq 的 setup.py")
setup_py.write_text(text.replace(needle, insert, 1), encoding="utf-8")
print("已跳过源码包里缺失的 C++ 扩展")
PY

python -m pip install --no-cache-dir --no-build-isolation "$work/fairseq-0.12.2"

python - <<'PY'
import re
import site
from pathlib import Path

pattern = re.compile(r"^(\s+)(\w+)\s*:\s*([A-Za-z_]\w*)\s*=\s*\3\(\)\s*$", re.M)

def repl(match):
    indent, name, typ = match.group(1), match.group(2), match.group(3)
    return f"{indent}{name}: {typ} = field(default_factory={typ})"

factory_pattern = re.compile(
    r"field\(\s*default\s*=\s*([A-Za-z_][\w.]*)\(\s*\)(\s*,)?"
)

def factory_repl(match):
    typ = match.group(1)
    comma = match.group(2) or ""
    return f"field(default_factory={typ}{comma}"

total = 0
for root in site.getsitepackages():
    base = Path(root)
    for pkg in ("fairseq", "hydra"):
        folder = base / pkg
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not re.search(r"from dataclasses import [^\n]*\bfield\b", text):
                continue
            patched, count = pattern.subn(repl, text)
            patched, count2 = factory_pattern.subn(factory_repl, patched)
            count += count2
            if count:
                path.write_text(patched, encoding="utf-8")
                print("已修补", path.name, count, "处")
                total += count
init_py = None
for root in site.getsitepackages():
    path = Path(root) / "fairseq" / "dataclass" / "initialize.py"
    if path.is_file():
        init_py = path
        break
if init_py is not None:
    init_text = init_py.read_text(encoding="utf-8")
    found = re.search(
        r"^([ \t]*)v = FairseqConfig\.__dataclass_fields__\[k\]\.default\s*$",
        init_text,
        re.M,
    )
    if found and "_field.default_factory" not in init_text:
        indent = found.group(1)
        init_text = init_text.replace(
            found.group(0),
            "\n".join(
                [
                    indent + "_field = FairseqConfig.__dataclass_fields__[k]",
                    indent + "v = _field.default",
                    indent + "if v is MISSING and _field.default_factory is not MISSING:",
                    indent + "    v = _field.default_factory()",
                ]
            ),
            1,
        )
        if "from dataclasses import MISSING" not in init_text:
            init_text = init_text.replace(
                "import logging\n",
                "import logging\nfrom dataclasses import MISSING\n",
                1,
            )
        init_py.write_text(init_text, encoding="utf-8")
        print("已修补 fairseq 启动逻辑")
print("Python 3.11 兼容修补完成，共", total, "处")
PY
