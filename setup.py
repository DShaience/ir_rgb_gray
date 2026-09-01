"""Set up a Python environment for ir_rgb_gray -- either a plain venv or a
conda env, so you can pick whichever you already have tooling for.

Usage:
    python setup.py                          interactive prompt
    python setup.py --env venv               create ./.venv and pip install into it
    python setup.py --env venv --venv-dir .venv2
    python setup.py --env conda              create/reuse a conda env and pip install into it
    python setup.py --env conda --name my_env --python 3.11
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def setup_venv(venv_dir: Path) -> None:
    if venv_dir.exists():
        print(f"{venv_dir} already exists, reusing it.")
    else:
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if platform.system() == "Windows":
        python_bin = venv_dir / "Scripts" / "python.exe"
        activate_hint = f"{venv_dir}\\Scripts\\activate"
    else:
        python_bin = venv_dir / "bin" / "python"
        activate_hint = f"source {venv_dir}/bin/activate"

    run([str(python_bin), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([str(python_bin), "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])

    print("\nDone. Activate with:")
    print(f"  {activate_hint}")


def setup_conda(env_name: str, python_version: str) -> None:
    if shutil.which("conda") is None:
        print("conda was not found on PATH. Install Miniconda/Anaconda first, or use --env venv instead.")
        sys.exit(1)

    existing = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
    names = {line.split()[0] for line in existing.stdout.splitlines() if line and not line.startswith("#")}
    if env_name in names:
        print(f"conda env '{env_name}' already exists, reusing it.")
    else:
        run(["conda", "create", "-y", "-n", env_name, f"python={python_version}"])

    run(["conda", "run", "-n", env_name, "python", "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])

    print("\nDone. Activate with:")
    print(f"  conda activate {env_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["venv", "conda"], help="which environment manager to use")
    parser.add_argument("--name", default="ir_rgb_gray", help="conda env name (--env conda only)")
    parser.add_argument("--python", default="3.11", help="python version for the conda env (--env conda only)")
    parser.add_argument("--venv-dir", default=".venv", help="venv directory, relative to this file (--env venv only)")
    args = parser.parse_args()

    env = args.env
    if env is None:
        choice = input("Set up with (v)env or (c)onda? [v/c]: ").strip().lower()
        env = "conda" if choice.startswith("c") else "venv"

    if env == "venv":
        setup_venv(ROOT / args.venv_dir)
    else:
        setup_conda(args.name, args.python)


if __name__ == "__main__":
    main()
