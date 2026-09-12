"""Package only known source directories; never include retinal datasets or runs."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="dist/CGDD-Net_GitHub.zip")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    top_names={"README.md","README_zh.md","CITATION.cff","RELEASE_NOTES.md","LICENSE","NOTICE","pyproject.toml","requirements.txt","train.py","evaluate.py",".gitignore"}
    directories={"LICENSES","cgddnet","configs","scripts","tests","docs","results","assets","splits",".github"}
    paths=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if any(part in {"__pycache__",".pytest_cache",".DS_Store"} for part in rel.parts): continue
        if p.suffix in {".pyc",".pt",".pth",".ckpt",".npz",".npy"}: continue
        if rel.parts[0] in directories or rel.as_posix() in top_names: paths.append(p)
    destination=Path(args.output).resolve(); destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(root).as_posix()+"\n" for p in paths)
    with ZipFile(destination,"w",ZIP_DEFLATED) as archive:
        for p in paths: archive.write(p,p.relative_to(root).as_posix())
        archive.writestr("SHA256SUMS.txt",manifest)
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for p in paths: assert archive.read(p.relative_to(root).as_posix())==p.read_bytes()
    print(f"{destination}: {len(paths)} source files; {destination.stat().st_size} bytes")


if __name__=="__main__": main()
