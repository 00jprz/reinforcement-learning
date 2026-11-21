"""
Proyecto del curso - Aprendizaje por refuerzo (Maestria en IA, Universidad de los Andes)
Entrega final: Q-learning tabular para llegar a la meta en un cuarto de 8 x 8 con 5 obstaculos moviles.

Integrantes: Javier Perez Osorio, Carlos Jose Garcia Corrales

Este archivo contiene TODO el codigo de la solucion:
  - Ambiente (estados, acciones, dinamica y recompensas definidos en la entrega parcial).
  - Estado observable s = (f, c, d); los obstaculos son internos al simulador.
  - Q-learning con politica epsilon-voraz.
  - Funciones para GUARDAR y CARGAR la Q-tabla (guardar_qtabla / cargar_qtabla).
  - Evaluacion de la politica y generacion del video de demostracion.

Uso (desde la terminal):
  python proyecto_rl.py entrenar            # entrena y guarda Q_tabla.json
  python proyecto_rl.py evaluar             # carga Q_tabla.json y mide el desempeno
  python proyecto_rl.py video               # genera demo_agente.mp4 (requiere matplotlib y ffmpeg)
  python proyecto_rl.py todo                # las tres cosas en orden

Dependencias: numpy (todo), matplotlib y ffmpeg (solo para el video).
"""
import csv
import json
import os
import random
import sys
import time

import numpy as np

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))


def _ruta_proyecto(nombre):
    """Devuelve una ruta dentro de la carpeta que contiene este archivo."""
    return os.path.join(CARPETA_PROYECTO, nombre)

# ----------------------------------------------------------------------------
# 1. Constantes del problema
# ----------------------------------------------------------------------------
N = 8                      # el cuarto es de N x N casillas
META = (7, 7)              # casilla meta (fila, columna)
INICIO = (0, 0)            # casilla inicial del agente
N_OBST = 5                 # numero de obstaculos
MAX_PASOS = 100            # limite de acciones por episodio

# Orientaciones: 0 = norte, 1 = este, 2 = sur, 3 = oeste
NOMBRES_DIR = ["norte", "este", "sur", "oeste"]
DESPL = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # (delta fila, delta columna) al avanzar
DIR_INICIAL = 1                              # este

# Acciones
G_IZQ, G_DER, AV = 0, 1, 2
NOMBRES_ACC = ["G_izq", "G_der", "Av"]

# Recompensas r(s, a, s')
R_COLISION = -100.0
R_META = +100.0
R_PASO = -1.0
GAMMA = 1.0

# Estados terminales
ACTIVO, S_META, S_COL = "activo", "s_meta", "s_col"
TRUNCADO = "truncado"       # resultado externo; no es un estado del ambiente


# ----------------------------------------------------------------------------
# 2. Ambiente
# ----------------------------------------------------------------------------
class Ambiente:
    """El agente observa s=(f,c,d); O permanece interno al simulador."""

    def __init__(self, semilla=None):
        self.rng = random.Random(semilla)
        self.reset()

    # -- inicio de episodio ---------------------------------------------------
    def reset(self):
        casillas = [(i, j) for i in range(N) for j in range(N) if (i, j) != INICIO]   # 63 casillas
        self.f, self.c = INICIO
        self.d = DIR_INICIAL
        self.obst = tuple(self.rng.sample(casillas, N_OBST))   # 5 casillas distintas, equiprobables
        self.estado = ACTIVO
        return self.observacion()

    # -- acciones aplicables A(s) ---------------------------------------------
    def acciones_aplicables(self):
        if self.estado != ACTIVO:
            return []
        df, dc = DESPL[self.d]
        if 0 <= self.f + df < N and 0 <= self.c + dc < N:
            return [G_IZQ, G_DER, AV]
        return [G_IZQ, G_DER]          # mirando hacia el borde: no se puede avanzar

    # -- estado observado por el agente s = (f, c, d) ----------------------------
    def observacion(self):
        if self.estado != ACTIVO:
            return None
        return (self.f, self.c, self.d)

    # -- movimiento aleatorio de los obstaculos ----------------------------------
    def _mover_obstaculos(self):
        opciones = []
        for (i, j) in self.obst:
            ops = [(i, j)]
            for df, dc in DESPL:
                if 0 <= i + df < N and 0 <= j + dc < N:
                    ops.append((i + df, j + dc))
            opciones.append(ops)
        while True:   # se repite el sorteo hasta que los 5 destinos sean distintos
            nuevas = tuple(self.rng.choice(ops) for ops in opciones)
            if len(set(nuevas)) == N_OBST:
                return nuevas

    # -- un paso del ambiente: devuelve (nueva observacion, recompensa, terminado) -----
    def paso(self, a):
        assert self.estado == ACTIVO, "episodio terminado"
        assert a in self.acciones_aplicables(), "accion no aplicable en este estado"
        f, c, d = self.f, self.c, self.d
        # 1) agente
        if a == G_IZQ:
            f2, c2, d2 = f, c, (d - 1) % 4
        elif a == G_DER:
            f2, c2, d2 = f, c, (d + 1) % 4
        else:
            f2, c2, d2 = f + DESPL[d][0], c + DESPL[d][1], d
        # 2) obstaculos (movimiento simultaneo)
        viejas = self.obst
        nuevas = self._mover_obstaculos()
        # 3) colision: misma casilla o intercambio de posiciones
        choque = (f2, c2) in nuevas
        if not choque and (f2, c2) != (f, c):
            for ant, nue in zip(viejas, nuevas):
                if ant == (f2, c2) and nue == (f, c):
                    choque = True
                    break
        self.f, self.c, self.d, self.obst = f2, c2, d2, nuevas
        if choque:
            self.estado, r = S_COL, R_COLISION
        elif (f2, c2) == META:
            self.estado, r = S_META, R_META
        else:
            r = R_PASO
        return self.observacion(), r, self.estado != ACTIVO


# ----------------------------------------------------------------------------
# 3. Q-tabla: diccionario {(f,c,d): [Q(G_izq), Q(G_der), Q(Av)]}
# ----------------------------------------------------------------------------
def _q_de(Q, obs):
    q = Q.get(obs)
    if q is None:
        q = [0.0, 0.0, 0.0]
        Q[obs] = q
    return q


def mejor_accion(Q, obs, aplicables, rng):
    """Accion voraz entre las aplicables; los empates se rompen al azar."""
    q = Q.get(obs, (0.0, 0.0, 0.0))
    mejor = max(q[a] for a in aplicables)
    return rng.choice([a for a in aplicables if q[a] == mejor])


def guardar_qtabla(Q, ruta=None):
    """Guarda la Q-tabla; por defecto sobrescribe 12-Proyecto/Q_tabla.json."""
    if ruta is None:
        ruta = _ruta_proyecto("Q_tabla.json")
    datos = {
        "meta": {
            "version_esquema": 2,
            "descripcion": "Q-tabla del proyecto de RL. Clave: 'f,c,d' con d en "
                           "{0 norte, 1 este, 2 sur, 3 oeste}. "
                           "Valor: [Q(G_izq), Q(G_der), Q(Av)].",
            "acciones": NOMBRES_ACC,
            "gamma": GAMMA,
            "n_entradas": len(Q),
        },
        "Q": {"%d,%d,%d" % estado: [round(v, 6) for v in q] for estado, q in Q.items()},
    }
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False, separators=(",", ":"))
    return ruta


def cargar_qtabla(ruta=None):
    """Carga una Q-tabla guardada con guardar_qtabla y la devuelve como diccionario."""
    if ruta is None:
        ruta = _ruta_proyecto("Q_tabla.json")
    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)
    Q = {}
    for clave, q in datos["Q"].items():
        f, c, d = (int(x) for x in clave.split(","))
        Q[(f, c, d)] = [float(v) for v in q]
    return Q


def exportar_csv(Q, ruta=None):
    """Version legible de la Q-tabla, ordenada por posicion y orientacion."""
    if ruta is None:
        ruta = _ruta_proyecto("Q_tabla.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fila", "columna", "orientacion", "Q_G_izq", "Q_G_der", "Q_Av", "mejor_accion"])
        for (f, c, d), q in sorted(Q.items()):
            aplic = [G_IZQ, G_DER, AV] if (0 <= f + DESPL[d][0] < N and 0 <= c + DESPL[d][1] < N) else [G_IZQ, G_DER]
            mejor = max(aplic, key=lambda a: q[a])
            w.writerow([f, c, NOMBRES_DIR[d], round(q[0], 3), round(q[1], 3), round(q[2], 3), NOMBRES_ACC[mejor]])
    return ruta


# ----------------------------------------------------------------------------
# 4. Q-learning
# ----------------------------------------------------------------------------
def entrenar(episodios=1_000_000, alpha=0.1, alpha_fin=0.01, eps_ini=1.0, eps_fin=0.01, frac_decay=0.8, semilla=0, verbose=True):
    """Q-learning tabular con politica epsilon-voraz; epsilon (y opcionalmente alpha) decrecen linealmente."""
    alpha_ini = alpha
    if alpha_fin is None:
        alpha_fin = alpha_ini
    env = Ambiente(semilla)
    rng = random.Random(semilla + 1)
    Q = {}
    historial = []   # (episodio, tasa de exito, retorno medio) por ventanas de 50000 episodios
    ventana = {"exito": 0, "ret": 0.0, "n": 0}
    t0 = time.time()
    n_decay = max(1, int(episodios * frac_decay))
    for ep in range(1, episodios + 1):
        eps = max(eps_fin, eps_ini - (eps_ini - eps_fin) * ep / n_decay)
        alpha = alpha_ini + (alpha_fin - alpha_ini) * ep / episodios
        estado = env.reset()
        ret = 0.0
        for _paso in range(1, MAX_PASOS + 1):
            aplic = env.acciones_aplicables()
            if rng.random() < eps:
                a = rng.choice(aplic)
            else:
                a = mejor_accion(Q, estado, aplic, rng)
            estado2, r, terminado = env.paso(a)
            q = _q_de(Q, estado)
            if terminado:
                objetivo = r                                   # el valor de un estado terminal es 0
            else:
                q2 = Q.get(estado2, (0.0, 0.0, 0.0))
                objetivo = r + GAMMA * max(q2[x] for x in env.acciones_aplicables())
            q[a] += alpha * (objetivo - q[a])
            estado = estado2
            ret += r
            if terminado:
                break
        ventana["exito"] += env.estado == S_META
        ventana["ret"] += ret
        ventana["n"] += 1
        if ep % 50000 == 0:
            historial.append((ep, ventana["exito"] / ventana["n"], ventana["ret"] / ventana["n"]))
            if verbose:
                print(f"  episodio {ep:>7d} | eps={eps:.3f} | exito (ventana, con exploracion)={historial[-1][1]:.3f} "
                      f"| retorno medio={historial[-1][2]:8.2f} | entradas Q={len(Q)} | {time.time()-t0:.0f}s", flush=True)
            ventana = {"exito": 0, "ret": 0.0, "n": 0}
    return Q, historial


# ----------------------------------------------------------------------------
# 5. Evaluacion
# ----------------------------------------------------------------------------
def evaluar(Q, episodios=5000, semilla=12345, politica="Q"):
    """Politica voraz (politica='Q') o aleatoria entre aplicables (politica='azar')."""
    env = Ambiente(semilla)
    rng = random.Random(semilla + 7)
    cuenta = {S_META: 0, S_COL: 0, TRUNCADO: 0}
    pasos_exito, retornos = [], []
    for _ in range(episodios):
        estado = env.reset()
        ret = 0.0
        resultado = TRUNCADO
        for paso in range(1, MAX_PASOS + 1):
            aplic = env.acciones_aplicables()
            a = mejor_accion(Q, estado, aplic, rng) if politica == "Q" else rng.choice(aplic)
            estado, r, terminado = env.paso(a)
            ret += r
            if terminado:
                resultado = env.estado
                break
        cuenta[resultado] += 1
        retornos.append(ret)
        if resultado == S_META:
            pasos_exito.append(paso)
    res = {
        "episodios": episodios,
        "tasa_exito": cuenta[S_META] / episodios,
        "tasa_colision": cuenta[S_COL] / episodios,
        "tasa_limite": cuenta[TRUNCADO] / episodios,
        "pasos_medios_exito": float(np.mean(pasos_exito)) if pasos_exito else float("nan"),
        "retorno_medio": float(np.mean(retornos)),
    }
    return res


def imprimir(titulo, r):
    print(f"{titulo}\n  episodios={r['episodios']} | exito={r['tasa_exito']*100:.1f}% | colision={r['tasa_colision']*100:.1f}% "
          f"| truncamiento={r['tasa_limite']*100:.1f}% | pasos medios (exitos)={r['pasos_medios_exito']:.1f} | retorno medio={r['retorno_medio']:.1f}")


# ----------------------------------------------------------------------------
# 6. Video de demostracion
# ----------------------------------------------------------------------------
def grabar_episodio(Q, semilla, rng_sel):
    """Ejecuta un episodio con la politica aprendida y devuelve la lista de cuadros para el video."""
    env = Ambiente(semilla)
    cuadros = [dict(f=env.f, c=env.c, d=env.d, obst=env.obst, t=0, acc=None, r=None, ret=0.0, estado=ACTIVO)]
    estado, ret = env.observacion(), 0.0
    for paso in range(1, MAX_PASOS + 1):
        aplic = env.acciones_aplicables()
        a = mejor_accion(Q, estado, aplic, rng_sel)
        estado, r, terminado = env.paso(a)
        ret += r
        resultado = env.estado if terminado else (TRUNCADO if paso == MAX_PASOS else ACTIVO)
        cuadros.append(dict(f=env.f, c=env.c, d=env.d, obst=env.obst, t=paso, acc=a, r=r, ret=ret, estado=resultado))
        if terminado:
            break
    return cuadros


def generar_video(Q, ruta=None, semillas=(3, 11, 21, 34, 55, 89), fps=3, carpeta_tmp=None):
    import shutil
    import subprocess
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    if ruta is None:
        ruta = _ruta_proyecto("demo_agente.mp4")
    if carpeta_tmp is None:
        carpeta_tmp = _ruta_proyecto("_frames")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "No se encontro ffmpeg. Instala las dependencias con "
                "'python -m pip install -r requirements.txt' o instala ffmpeg "
                "en el sistema."
            ) from exc
    os.makedirs(carpeta_tmp, exist_ok=True)
    for f_ in os.listdir(carpeta_tmp):
        os.remove(os.path.join(carpeta_tmp, f_))

    rng_sel = random.Random(2024)
    episodios = [grabar_episodio(Q, s, rng_sel) for s in semillas]
    resultados = {S_META: 0, S_COL: 0, TRUNCADO: 0}
    k = 0
    AZUL, ROJO, VERDE, GRIS = "#1f5f8b", "#c0392b", "#27ae60", "#9aa5ad"
    # triangulo orientado: puntos en coordenadas locales mirando al norte
    base = np.array([[0, -0.38], [0.30, 0.30], [-0.30, 0.30]])
    giro = {0: 0, 1: 90, 2: 180, 3: 270}

    def triangulo(f, c, d):
        ang = np.deg2rad(giro[d])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = base @ R.T
        return pts + np.array([c + 0.5, f + 0.5])

    def dibujar(cuadro, num_ep, total, resumen, pausa_texto=""):
        nonlocal k
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_axes([0.04, 0.06, 0.56, 0.88])
        for i in range(N):
            for j in range(N):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#f4f7f9" if (i + j) % 2 == 0 else "#e6edf1", edgecolor="#cdd8e0", lw=1))
        ax.add_patch(Rectangle((META[1], META[0]), 1, 1, facecolor="#d5f0df", edgecolor=VERDE, lw=2))
        ax.text(META[1] + .5, META[0] + .5, "META", ha="center", va="center", color=VERDE, fontsize=12, weight="bold")
        for (i, j) in cuadro["obst"]:
            ax.add_patch(Rectangle((j + .12, i + .12), .76, .76, facecolor=ROJO, edgecolor="#7b241c", lw=1.5))
        ax.add_patch(Polygon(triangulo(cuadro["f"], cuadro["c"], cuadro["d"]), closed=True, facecolor=AZUL, edgecolor="#0b2e45", lw=1.5, zorder=5))
        if cuadro["estado"] == S_COL:
            ax.text(cuadro["c"] + .5, cuadro["f"] - .12, "¡COLISIÓN!", ha="center", color=ROJO, fontsize=13, weight="bold", zorder=6)
        ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal")
        ax.set_xticks(np.arange(N) + .5); ax.set_xticklabels(range(N), fontsize=9)
        ax.set_yticks(np.arange(N) + .5); ax.set_yticklabels(range(N), fontsize=9)
        ax.tick_params(length=0)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_title(f"Episodio {num_ep} de {total}", fontsize=15, weight="bold", loc="left", color="#163d59")

        panel = fig.add_axes([0.63, 0.06, 0.35, 0.88]); panel.axis("off")
        panel.set_xlim(0, 1); panel.set_ylim(0, 1)
        panel.text(0, 0.97, "Agente Q-learning", fontsize=17, weight="bold", color="#163d59", va="top")
        panel.text(0, 0.915, "Proyecto RL · Uniandes", fontsize=10.5, color="#586774", va="top")
        acc = "—" if cuadro["acc"] is None else NOMBRES_ACC[cuadro["acc"]]
        lineas = [
            ("Paso (t)", f"{cuadro['t']} / {MAX_PASOS}"),
            ("Posición (f, c)", f"({cuadro['f']}, {cuadro['c']})"),
            ("Orientación", NOMBRES_DIR[cuadro["d"]]),
            ("Última acción", acc),
            ("Recompensa del paso", "—" if cuadro["r"] is None else f"{cuadro['r']:+.0f}"),
            ("Retorno acumulado", f"{cuadro['ret']:+.0f}"),
        ]
        y = 0.80
        for et, val in lineas:
            panel.text(0, y, et, fontsize=12, color="#586774", va="top")
            panel.text(1, y, val, fontsize=13, weight="bold", ha="right", va="top", color="#0b2e45")
            y -= 0.07
        est = cuadro["estado"]
        if est == S_META: txt, col = "META ALCANZADA", VERDE
        elif est == S_COL: txt, col = "COLISIÓN", ROJO
        elif est == TRUNCADO: txt, col = "EPISODIO TRUNCADO", "#b9770e"
        else: txt, col = "en curso…", "#586774"
        panel.text(0, 0.35, "Estado del episodio", fontsize=12, color="#586774", va="top")
        panel.text(0, 0.30, txt, fontsize=17, weight="bold", color=col, va="top")
        panel.text(0, 0.19, "Acumulado de la demostración", fontsize=12, color="#586774", va="top")
        panel.text(0, 0.145, f"Metas: {resumen[S_META]}   Colisiones: {resumen[S_COL]}   Truncados: {resumen[TRUNCADO]}",
                   fontsize=12, weight="bold", color="#0b2e45", va="top")
        panel.text(0, 0.03, "Azul: agente (el vértice indica la orientación)\nRojo: obstáculos móviles · Verde: meta", fontsize=9.5, color="#586774", va="bottom")
        fig.savefig(os.path.join(carpeta_tmp, f"f{k:05d}.png"), facecolor="white")
        plt.close(fig)
        k += 1

    # tarjeta de titulo
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); fig.patch.set_facecolor("#163d59")
    fig.text(0.5, 0.62, "Navegación con obstáculos móviles", ha="center", color="white", fontsize=34, weight="bold")
    fig.text(0.5, 0.52, "Q-learning tabular · Entrega final del proyecto", ha="center", color="#cfe3f1", fontsize=20)
    fig.text(0.5, 0.36, "Javier Pérez Osorio · Carlos José García Corrales", ha="center", color="white", fontsize=18)
    fig.text(0.5, 0.30, "Maestría en Inteligencia Artificial · Universidad de los Andes", ha="center", color="#cfe3f1", fontsize=14)
    for _ in range(fps * 3):
        fig.savefig(os.path.join(carpeta_tmp, f"f{k:05d}.png"), facecolor=fig.get_facecolor()); k += 1
    plt.close(fig)

    total = len(episodios)
    for e, cuadros in enumerate(episodios, 1):
        for idx, cu in enumerate(cuadros):
            dibujar(cu, e, total, resultados)
        final = cuadros[-1]
        resultados[final["estado"]] += 1
        for _ in range(fps * 2):          # pausa de 2 s con el resultado
            dibujar(final, e, total, resultados)

    # Codificar fuera de carpetas sincronizadas evita timeouts del proveedor
    # (por ejemplo, OneDrive) mientras ffmpeg escribe y cierra el contenedor.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as archivo_tmp:
        ruta_tmp = archivo_tmp.name
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(carpeta_tmp, "f%05d.png"),
                        "-vf", "scale=1280:720", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", ruta_tmp], check=True)
        shutil.copyfile(ruta_tmp, ruta)
    finally:
        if os.path.exists(ruta_tmp):
            os.remove(ruta_tmp)
    shutil.rmtree(carpeta_tmp)
    return ruta, resultados


# ----------------------------------------------------------------------------
# 7. Programa principal
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    orden = sys.argv[1] if len(sys.argv) > 1 else "todo"
    RUTA_Q = _ruta_proyecto("Q_tabla.json")
    RUTA_CSV = _ruta_proyecto("Q_tabla.csv")
    if orden in ("entrenar", "todo"):
        print("Entrenando…")
        Q, hist = entrenar()
        guardar_qtabla(Q, RUTA_Q)
        exportar_csv(Q, RUTA_CSV)
        print(f"Q-tabla guardada en {RUTA_Q} ({len(Q)} estados visitados).")
    if orden in ("evaluar", "todo"):
        Q = cargar_qtabla(RUTA_Q)
        imprimir("Politica aprendida (voraz):", evaluar(Q, politica="Q"))
        imprimir("Politica aleatoria (referencia):", evaluar(Q, politica="azar"))
    if orden in ("video", "todo"):
        Q = cargar_qtabla(RUTA_Q)
        ruta, res = generar_video(Q)
        print(f"Video guardado en {ruta}. Resultados de la demostracion: {res}")
