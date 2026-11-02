"""Prepara el único entorno compartido del repositorio: .venv en la raíz."""

from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


def setup() -> None:
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-cache-dir", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
    subprocess.run(
        [
            str(python), "-m", "ipykernel", "install",
            "--prefix", str(environment),
            "--name", "reinforcement-learning",
            "--display-name", "Python (reinforcement-learning)",
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print(f"Entorno compartido listo: {environment}", flush=True)


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Se requiere Python 3.10 o posterior; validado con Python 3.14.")
    if sys.argv[1:]:
        raise SystemExit("Ejecuta python3 setup_environments.py sin carpetas: el entorno es compartido.")
    setup()


if __name__ == "__main__":
    main()
