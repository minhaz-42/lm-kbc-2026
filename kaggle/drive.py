"""Drive a Kaggle kernel end-to-end via the API: init metadata, push, wait, pull.

Prereq: your Kaggle API token at ~/.kaggle/kaggle.json
  (Kaggle -> Settings -> API -> "Create New Token" downloads it).

    python kaggle/drive.py init     # write kernel-metadata.json (reads your username)
    python kaggle/drive.py push     # upload + start the kernel (GPU + internet)
    python kaggle/drive.py wait      # poll until complete/error  (run in background)
    python kaggle/drive.py pull      # download outputs to kaggle/output/
    python kaggle/drive.py run       # push + wait + pull in one go
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = "kernel.ipynb"
SLUG = "lm-kbc-2026-val"
KAGGLE = os.path.expanduser("~/.kaggle/kaggle.json")
OUTDIR = os.path.join(HERE, "output")


def username():
    # works with either auth method: parse `kaggle config view`
    r = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if "username:" in line:
            u = line.split("username:")[1].strip()
            if u and u.lower() != "none":
                return u
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if os.path.exists(KAGGLE):
        return json.load(open(KAGGLE))["username"]
    sys.exit("could not determine Kaggle username; set KAGGLE_USERNAME or ~/.kaggle config")


def kaggle(*args, capture=True):
    return subprocess.run(["kaggle", *args], capture_output=capture, text=True)


def kid():
    return f"{username()}/{SLUG}"


def cmd_init():
    meta = {
        "id": kid(),
        "title": SLUG,
        "code_file": NB,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    path = os.path.join(HERE, "kernel-metadata.json")
    json.dump(meta, open(path, "w"), indent=2)
    print("wrote", path, "->", meta["id"])


def cmd_push():
    if not os.path.exists(os.path.join(HERE, "kernel-metadata.json")):
        cmd_init()
    r = kaggle("kernels", "push", "-p", HERE)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)


def cmd_status():
    r = kaggle("kernels", "status", kid())
    out = (r.stdout + r.stderr).strip()
    print(out)
    return out.lower()


def cmd_wait(timeout=5400, every=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = cmd_status()
        if "complete" in s:
            print("== COMPLETE =="); return 0
        if "error" in s or "cancel" in s:
            print("== FAILED =="); return 1
        time.sleep(every)
    print("== TIMEOUT =="); return 2


def cmd_pull():
    os.makedirs(OUTDIR, exist_ok=True)
    r = kaggle("kernels", "output", kid(), "-p", OUTDIR)
    print(r.stdout, r.stderr)
    print("files:", sorted(os.listdir(OUTDIR)))


def cmd_run():
    cmd_push()
    if cmd_wait() == 0:
        cmd_pull()


if __name__ == "__main__":
    {"init": cmd_init, "push": cmd_push, "status": cmd_status,
     "wait": cmd_wait, "pull": cmd_pull, "run": cmd_run}[sys.argv[1]]()
