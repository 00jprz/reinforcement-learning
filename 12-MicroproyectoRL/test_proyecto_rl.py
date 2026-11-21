import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULO = Path(__file__).with_name("proyecto_rl.py")
SPEC = importlib.util.spec_from_file_location("proyecto_rl", MODULO)
rl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rl)


class AmbienteTests(unittest.TestCase):
    def test_reset_devuelve_solo_posicion_y_orientacion(self):
        env = rl.Ambiente(semilla=1)
        self.assertEqual(env.observacion(), (0, 0, rl.DIR_INICIAL))
        self.assertEqual(len(env.observacion()), 3)
        self.assertFalse(hasattr(env, "t"))
        self.assertNotIn(rl.INICIO, env.obst)
        self.assertNotIn(rl.META, env.obst)
        self.assertEqual(len(set(env.obst)), rl.N_OBST)

    def test_obstaculos_siempre_ocupan_casillas_validas_y_distintas(self):
        env = rl.Ambiente(semilla=11)
        for _ in range(200):
            env.obst = env._mover_obstaculos()
            self.assertEqual(len(env.obst), rl.N_OBST)
            self.assertEqual(len(set(env.obst)), rl.N_OBST)
            self.assertNotIn(rl.INICIO, env.obst)
            self.assertNotIn(rl.META, env.obst)
            self.assertTrue(all(casilla in rl.CASILLAS_OBSTACULOS for casilla in env.obst))

    def test_giros_y_avance(self):
        env = rl.Ambiente(semilla=2)
        env._mover_obstaculos = lambda: env.obst
        obs, recompensa, terminado = env.paso(rl.G_DER)
        self.assertEqual(obs, (0, 0, 2))
        self.assertEqual(recompensa, rl.R_PASO)
        self.assertFalse(terminado)
        obs, _, _ = env.paso(rl.AV)
        self.assertEqual(obs, (1, 0, 2))

    def test_no_avanza_fuera_del_tablero(self):
        env = rl.Ambiente(semilla=3)
        env.d = 0
        self.assertNotIn(rl.AV, env.acciones_aplicables())
        with self.assertRaises(AssertionError):
            env.paso(rl.AV)

    def test_colision_tiene_prioridad_y_recompensa_negativa(self):
        env = rl.Ambiente(semilla=4)
        env.f, env.c, env.d = 0, 0, 1
        env.obst = ((0, 1), (2, 2), (3, 3), (4, 4), (5, 5))
        env._mover_obstaculos = lambda: env.obst
        obs, recompensa, terminado = env.paso(rl.AV)
        self.assertIsNone(obs)
        self.assertEqual(recompensa, rl.R_COLISION)
        self.assertTrue(terminado)
        self.assertEqual(env.estado, rl.S_COL)

    def test_meta_tiene_recompensa_positiva(self):
        env = rl.Ambiente(semilla=5)
        env.f, env.c, env.d = 7, 6, 1
        env.obst = ((0, 1), (1, 1), (2, 2), (3, 3), (4, 4))
        env._mover_obstaculos = lambda: env.obst
        obs, recompensa, terminado = env.paso(rl.AV)
        self.assertIsNone(obs)
        self.assertEqual(recompensa, rl.R_META)
        self.assertTrue(terminado)
        self.assertEqual(env.estado, rl.S_META)

    def test_ambiente_no_tiene_limite_de_pasos(self):
        env = rl.Ambiente(semilla=6)
        env._mover_obstaculos = lambda: env.obst
        for _ in range(101):
            obs, recompensa, terminado = env.paso(rl.G_IZQ)
            self.assertEqual(recompensa, rl.R_PASO)
            self.assertFalse(terminado)
            self.assertIsNotNone(obs)
        self.assertEqual(env.estado, rl.ACTIVO)


class PersistenciaTests(unittest.TestCase):
    def test_json_nuevo_hace_roundtrip(self):
        Q = {(0, 0, 1): [1.25, -2.5, 3.75]}
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "q.json"
            rl.guardar_qtabla(Q, ruta)
            with ruta.open(encoding="utf-8") as fh:
                datos = json.load(fh)
            self.assertEqual(datos["meta"]["version_esquema"], rl.VERSION_ESQUEMA_Q)
            self.assertIn("0,0,1", datos["Q"])
            self.assertEqual(rl.cargar_qtabla(ruta), Q)

    def test_json_antiguo_exige_reentrenamiento(self):
        antiguo = {
            "meta": {"descripcion": "esquema anterior"},
            "Q": {"0,0,1,0": [0.0, 0.0, 0.0]},
        }
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "q_antigua.json"
            ruta.write_text(json.dumps(antiguo), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Vuelva a entrenar"):
                rl.cargar_qtabla(ruta)

    def test_csv_no_incluye_sensor_de_obstaculos(self):
        Q = {(0, 0, 1): [1.0, 2.0, 3.0]}
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "q.csv"
            rl.exportar_csv(Q, ruta)
            with ruta.open(newline="", encoding="utf-8") as fh:
                filas = list(csv.reader(fh))
            self.assertEqual(
                filas[0],
                ["fila", "columna", "orientacion", "Q_G_izq", "Q_G_der", "Q_Av", "mejor_accion"],
            )
            self.assertEqual(len(filas[1]), 7)


class CiclosTests(unittest.TestCase):
    def test_resuelve_ffmpeg_incluido_en_dependencias(self):
        ejecutable = rl._resolver_ffmpeg()
        self.assertTrue(os.path.isfile(ejecutable))
        self.assertTrue(os.access(ejecutable, os.X_OK))

    def test_evaluacion_termina_solo_por_meta_o_colision(self):
        Q = {}
        for c in range(7):
            Q[(0, c, 1)] = [0.0, 0.0, 10.0]
        Q[(0, 7, 1)] = [0.0, 10.0, 0.0]
        for f in range(7):
            Q[(f, 7, 2)] = [0.0, 0.0, 10.0]

        def reset_seguro(env):
            env.f, env.c, env.d = 0, 0, rl.DIR_INICIAL
            env.obst = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
            env.estado = rl.ACTIVO
            return env.observacion()

        with (
            patch.object(rl.Ambiente, "reset", reset_seguro),
            patch.object(rl.Ambiente, "_mover_obstaculos", lambda self: self.obst),
        ):
            resultado = rl.evaluar(Q, episodios=3, semilla=10)
        self.assertEqual(resultado["tasa_exito"], 1.0)
        self.assertEqual(resultado["tasa_colision"], 0.0)
        self.assertNotIn("tasa_corte", resultado)
        self.assertEqual(resultado["pasos_medios_exito"], 15.0)
        self.assertEqual(resultado["retorno_medio"], 86.0)

    def test_resumen_de_evaluacion_agrega_varias_semillas(self):
        resultados = [
            {
                "episodios": 10,
                "tasa_exito": 0.4,
                "tasa_colision": 0.6,
                "pasos_medios_exito": 15.0,
                "retorno_medio": -20.0,
            },
            {
                "episodios": 10,
                "tasa_exito": 0.6,
                "tasa_colision": 0.4,
                "pasos_medios_exito": 17.0,
                "retorno_medio": 0.0,
            },
        ]
        with patch.object(rl, "evaluar", side_effect=resultados):
            resumen = rl.evaluar_semillas({}, semillas=(1, 2), episodios_por_semilla=10)
        self.assertAlmostEqual(resumen["promedio"]["tasa_exito"], 0.5)
        self.assertAlmostEqual(resumen["desviacion"]["tasa_exito"], 0.1)
        self.assertEqual(resumen["promedio"]["pasos_medios_exito"], 16.0)

    def test_entrenamiento_corto_finaliza_y_usa_claves_de_tres_componentes(self):
        with patch.object(rl.Ambiente, "_mover_obstaculos", lambda self: self.obst):
            Q, historial = rl.entrenar(episodios=3, semilla=20, verbose=False)
        self.assertEqual(historial, [])
        self.assertTrue(Q)
        self.assertTrue(all(len(clave) == 3 for clave in Q))
        self.assertLessEqual(len(Q), rl.N * rl.N * 4)


if __name__ == "__main__":
    unittest.main()
