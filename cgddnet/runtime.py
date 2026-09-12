"""Portable configuration, deterministic seeding and run provenance."""
from pathlib import Path
import hashlib
import json
import platform
import random
import subprocess
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_name(name):
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu for a smoke test")
    return torch.device(name)


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ("model","preprocessing","training","inference"):
        if key not in config: raise ValueError(f"Missing configuration section: {key}")
    return config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(device):
    try:
        revision = subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
        dirty = bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except (subprocess.CalledProcessError,FileNotFoundError):
        revision, dirty = None, None
    return {"python":platform.python_version(),"platform":platform.platform(),"torch":str(torch.__version__),
            "cuda_runtime":torch.version.cuda,"device":str(device),
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "git_commit":revision,"git_dirty":dirty}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def capture_rng():
    state=np.random.get_state()
    return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python":random.getstate(),"numpy":[state[0],state[1].tolist(),state[2],state[3],state[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]: torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    random.setstate(state["python"])
    ns=state["numpy"]; np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))
