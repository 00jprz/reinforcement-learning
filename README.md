# Reinforcement learning

Todos los laboratorios comparten un único entorno virtual, `.venv`, en la raíz del repositorio y un único archivo de dependencias, `requirements.txt`. El entorno se ha verificado con Python 3.14 en macOS ARM64.

```text
reinforcement-learning/
├── .venv/                    # Entorno compartido (excluido de Git)
├── requirements.txt          # Dependencias para todos los laboratorios
├── setup_environments.py     # Prepara el entorno y su kernel
├── README.md
├── 1-Bandits/
├── 2-Agent-K/
├── 3-K-Bandidos/
├── 4-Gridworld_MDPs/
└── 5-MDP_assignment/
```

Las carpetas de laboratorios contienen sus notebooks y recursos. No necesitan entornos ni archivos de dependencias propios. El nombre del entorno es `.venv`; `.env` suele utilizarse para archivos de variables de configuración.

## Preparar el entorno compartido

Desde la raíz del repositorio:

```sh
python3 setup_environments.py
```

En Windows puedes usar `py setup_environments.py`. El script crea `.venv` si hace falta, instala el único `requirements.txt`, registra el kernel **Python (reinforcement-learning)** dentro del entorno y comprueba las dependencias. No recibe nombres de carpetas. La instalación de paquetes requiere acceso a PyPI.

Si solo necesitas instalar o actualizar las dependencias en un entorno ya creado:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

En Windows PowerShell, activa el entorno con:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

## Abrir cualquier laboratorio

Inicia JupyterLab desde la raíz con el entorno activo:

```sh
source .venv/bin/activate
python -m jupyterlab
```

Abre el notebook de la carpeta que quieras y selecciona **Python (reinforcement-learning)**. En VS Code selecciona el intérprete `.venv/bin/python` de la raíz, o `.venv\Scripts\python.exe` en Windows, mediante **Select Kernel → Python Environments**.

También puedes trabajar desde una subcarpeta: en macOS/Linux activa primero `source ../.venv/bin/activate`. Todos los laboratorios usarán los mismos paquetes. Para salir del entorno ejecuta `deactivate`.

## Dependencias y mantenimiento

`requirements.txt` incluye JupyterLab, ipykernel y matplotlib para los notebooks actuales. `random`, `math` y `functools` forman parte de Python y no se instalan con pip.

Cuando un laboratorio necesite otra biblioteca, añádela al `requirements.txt` de la raíz y vuelve a ejecutar `python -m pip install -r requirements.txt` con el entorno compartido activo. No crees archivos requirements ni entornos adicionales dentro de los laboratorios.

Si mueves el repositorio o cambias de versión de Python, recrea `.venv` con el script: los entornos contienen rutas absolutas y no se deben copiar entre equipos. La preparación del entorno no ejecuta ni completa los ejercicios de los notebooks.
