"""Clone repo CSDI ke third_party/CSDI pada commit yang dikunci, lalu verifikasi lisensi."""
import subprocess
import sys
from pathlib import Path

PIN = "7f24a436f08d98853a6b43d4f7f04e5a65ecdf27"
URL = "https://github.com/ermongroup/CSDI"
ROOT = Path(__file__).resolve().parents[1]
dest = ROOT / "third_party" / "CSDI"


def git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()


if not dest.exists():
    dest.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", URL, str(dest)], check=True)
git("-C", str(dest), "checkout", PIN)
head = git("-C", str(dest), "rev-parse", "HEAD")
assert head == PIN, f"HEAD {head} != {PIN}"
lic = (dest / "LICENSE").read_text().splitlines()[0]
note = f"repo: {URL}\ncommit: {head}\nlisensi: {lic}\n"
(ROOT / "third_party" / "CSDI_PROVENANCE.txt").write_text(note)
print(note)
print("OK. Lanjut: python scripts/csdi_benchmark.py")
