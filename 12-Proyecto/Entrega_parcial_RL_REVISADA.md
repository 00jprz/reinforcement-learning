# Entrega parcial: definición del ambiente, las acciones y las recompensas

Proyecto del curso · Aprendizaje por refuerzo · Maestría en Inteligencia Artificial, Universidad de los Andes

**Integrantes:** Javier Pérez Osorio · Carlos José García Corrales

**Resumen.** Este documento define los tres elementos básicos del problema de aprendizaje por refuerzo del proyecto: el conjunto de estados, el conjunto de acciones del agente y el modelo de recompensas. El problema es el de un agente que debe ir de la esquina superior izquierda a la inferior derecha de un cuarto de 8 × 8 sin chocar con cinco obstáculos que se mueven al azar. Todas las reglas están escritas de forma que puedan implementarse directamente en el código de la entrega final.

## 1. Descripción del problema y supuestos

El ambiente es un cuarto vacío de 8 × 8 casillas. El agente empieza siempre en la posición (0,0), arriba a la izquierda, y debe llegar a la meta en (7,7), abajo a la derecha. Dentro del cuarto hay 5 obstáculos que cambian de posición al azar en cada paso, al mismo tiempo que el agente decide su acción. Si el agente choca con un obstáculo, el episodio termina. El agente puede girar 90° a la izquierda o a la derecha, o avanzar a la casilla que tiene enfrente si esta existe.

La tabla separa lo que fija el enunciado de lo que decidimos nosotros, porque el enunciado deja varios detalles abiertos.

|**Fijado por el enunciado**|**Decisión nuestra (propuesta)**|
|---|---|
|Cuarto de 8 × 8 sin paredes internas.|Orientación inicial del agente: este.|
|Agente inicia en (0,0); meta en (7,7).|Posiciones iniciales de los obstáculos: 5 casillas distintas<br>elegidas al azar, con la misma probabilidad, entre las 63 casillas<br>distintas de (0,0).|
|Cinco obstáculos que se mueven al azar en cada<br>paso, al mismo tiempo que el agente actúa.|Movimiento de cada obstáculo: se queda o se mueve a una<br>casilla vecina (arriba, abajo, izquierda o derecha) dentro del<br>cuarto, con igual probabilidad; los cinco destinos deben ser<br>distintos.|
|Acciones del agente: girar 90° a la izquierda, girar 90°<br>a la derecha o avanzar a la casilla de enfrente si es<br>posible.|Colisión: el agente y un obstáculo terminan en la misma casilla<br>o se cruzan intercambiando posiciones.|
|El episodio termina cuando el agente choca con un<br>obstáculo.|El episodio también termina al llegar a la meta. Durante el entrenamiento puede truncarse externamente después de 100 acciones.|
|El agente debe llegar a la meta sin chocar.|Valores de recompensa (sección 5) y factor de descuento γ = 1.|



**Convención de coordenadas.** Una casilla se escribe (f, c), con f la fila y c la columna, ambas de 0 a 7. La fila aumenta hacia el sur (hacia abajo) y la columna aumenta hacia el este (hacia la derecha).

## 2. Conjunto de estados

### 2.1 Estado que percibe el agente

El estado utilizado por el agente y por la Q-tabla es la tupla

$$
s = (f,c,d).
$$

|**Variable**|**Tipo y rango**|**Significado**|
|---|---|---|
|$f$|Entero de 0 a 7|Fila que ocupa el agente.|
|$c$|Entero de 0 a 7|Columna que ocupa el agente.|
|$d$|Categórica: norte, este, sur, oeste|Orientación del agente; determina la casilla hacia la que avanza.|

Estas tres variables contienen las características que el agente necesita para aprender a desplazarse por el cuarto: su ubicación y la dirección en la que está orientado. El agente no dispone de un mapa ni de sensores de proximidad para detectar obstáculos. Solo conoce una colisión cuando esta ocurre y recibe la recompensa correspondiente.

### 2.2 Información interna del simulador

El simulador mantiene por separado el conjunto $O$ con las posiciones de los cinco obstáculos. Esta información se usa para moverlos y detectar colisiones, pero no forma parte del estado observado por el agente ni del índice de la Q-tabla. De este modo, los obstáculos pertenecen a la dinámica aleatoria del ambiente.

La meta $(7,7)$, el tamaño del cuarto y las reglas de movimiento son constantes del ambiente, por lo que tampoco se incluyen en $s$. Si el entrenamiento establece un máximo de 100 acciones, ese valor funciona como un criterio externo de truncamiento y no como una variable de estado.

### 2.3 Estados iniciales y terminales

**Estado inicial.** El agente comienza en $s_0 = (0,0,\text{este})$. El simulador ubica los cinco obstáculos al azar en casillas distintas de $(0,0)$.

**Estados terminales.** Hay dos resultados terminales: $s_{\text{meta}}$, cuando el agente llega a $(7,7)$ sin colisión, y $s_{\text{col}}$, cuando choca con un obstáculo. Desde un estado terminal no hay acciones aplicables. Un episodio interrumpido por alcanzar el máximo externo de acciones se considera truncado, no un tercer estado terminal del problema.

### 2.4 Tamaño del espacio de estados

Hay 64 posiciones y 4 orientaciones, por lo que el agente puede observar

$$
|S| = 64 \cdot 4 = 256
$$

estados no terminales, además de los dos resultados terminales. Como existen como máximo tres acciones por estado, una Q-tabla densa requiere a lo sumo $256 \cdot 3 = 768$ valores. Este tamaño permite aplicar Q-learning tabular directamente.

## 3. Conjunto de acciones

### 3.1 Acciones del agente

|**Acción**|**Efecto sobre el agente**|
|---|---|
|**Girar a la izquierda**(G_izq)|Rota 90° a la izquierda y conserva su casilla.|
|**Girar a la derecha**(G_der)|Rota 90° a la derecha y conserva su casilla.|
|**Avanzar**(Av)|Se desplaza una casilla en la dirección de su orientación y conserva la<br>orientación.|



### 3.2 Orientaciones

Cada orientación está asociada a un desplazamiento (Δf, Δc) y a un resultado de cada giro.

|**Orientación d**|**Desplazamiento al avanzar (Δf, Δc)**|**Tras G_izq queda en**|**Tras G_der queda en**|
|---|---|---|---|
|norte|(−1, 0)|oeste|este|
|este|(0, +1)|norte|sur|
|sur|(+1, 0)|este|oeste|
|oeste|(0, −1)|sur|norte|



### 3.3 Acciones aplicables en cada estado

La casilla de enfrente de un estado activo es (f + Δf, c + Δc), con (Δf, Δc) el desplazamiento de la orientación _d_ . Está dentro del tablero si ambas coordenadas están entre 0 y 7. El conjunto de acciones aplicables A(s) depende solo del tipo de estado.

|**Tipo de estado**|**Condición**|**Acciones aplicables A(s)**|
|---|---|---|
|Activo, con casilla de enfrente dentro del tablero|La casilla $(f + \Delta f, c + \Delta c)$ existe|$\{G_{\text{izq}}, G_{\text{der}}, Av\}$|
|Activo, mirando hacia el borde|Norte con $f = 0$, sur con $f = 7$, este con $c = 7$ u oeste con $c = 0$|$\{G_{\text{izq}}, G_{\text{der}}\}$|
|Terminal|$s_{\text{meta}}$ o $s_{\text{col}}$|$\varnothing$ (ninguna)|



Dos aclaraciones. Primero, Av es aplicable aunque la casilla de enfrente esté ocupada por un obstáculo al inicio del paso, porque ese obstáculo podría moverse; el resultado se decide en la dinámica de la sección 4. Segundo, estas son exactamente las acciones del enunciado: el agente solo puede rotar 90° o moverse a la casilla de enfrente si es posible, y no puede moverse hacia atrás ni en diagonal.

## 4. Dinámica del ambiente

En cada paso, la acción del agente y el movimiento de los obstáculos se calculan a partir de la configuración actual y se aplican a la vez. Dado un estado del agente $s = (f,c,d)$, la configuración interna $O$ del simulador y una acción $a \in A(s)$, el siguiente estado se obtiene así:

- **Agente.** Se calcula su nueva posición y orientación (fʹ, cʹ, dʹ) según la acción (sección 3).

- **Obstáculos.** Cada uno elige su destino con igual probabilidad entre quedarse y moverse a una casilla vecina dentro del tablero. Si dos o más eligen la misma casilla, se vuelve a sortear el movimiento de todos hasta que los cinco destinos sean distintos. El resultado es Oʹ.

- **Colisión.** Hay colisión si $(f',c')$ está en $O'$, o si algún obstáculo pasa de $(f',c')$ a $(f,c)$ mientras el agente pasa de $(f,c)$ a $(f',c')$ (intercambio). El resultado es $s_{\text{col}}$.

- **Meta.** Si no hubo colisión y $(f',c') = (7,7)$, el resultado es $s_{\text{meta}}$.

- **Truncamiento externo.** Después de procesar la transición, el ciclo de entrenamiento puede detener el episodio al alcanzar 100 acciones. Este control no modifica $s$ ni crea un estado terminal adicional.

- **Continuación.** En cualquier otro caso, el agente observa $s' = (f',c',d')$ y el simulador conserva internamente $O'$ para el siguiente paso.

El orden importa: la colisión se evalúa antes que la meta; si el agente llega a $(7,7)$ y un obstáculo está o llega allí, el resultado es una colisión. Para el agente, el movimiento no observado de los obstáculos se incorpora como aleatoriedad en las transiciones entre los 256 estados de su Q-tabla. En seudocódigo:

```text
paso(s, O, a):                            # s = (f, c, d), O es interno
    (f2, c2, d2) = aplicar(a, f, c, d)    # giro o avance
    O2 = mover_obstaculos(O)              # 5 destinos distintos
    si (f2, c2) en O2 o hay_intercambio(...):   retornar s_col, -100
    si (f2, c2) == (7, 7):                      retornar s_meta, +100
    retornar (f2, c2, d2), O2, -1
```

## 5. Modelo de recompensas

### 5.1 Recompensa sobre tripletas $(s, a, s')$

Como el mismo par $(s,a)$ puede llevar a resultados distintos por el movimiento aleatorio de los obstáculos, la recompensa se define sobre las tripletas $(s,a,s')$: $s$ es el estado de salida, $a$ la acción ejecutada y $s'$ el estado o resultado de llegada. La recompensa depende de si el episodio continúa, el agente alcanza la meta o se produce una colisión.

|**Tripleta $(s,a,s')$**|**Situación**|**Recompensa $r(s,a,s')$**|
|---|---|---|
|$s' = s_{\text{col}}$, para cualquier $s$ activo y $a \in A(s)$|El agente choca con un obstáculo, incluso si en ese mismo paso llegaba a la meta.|$-100$|
|$s' = s_{\text{meta}}$, para cualquier $s$ activo y $a = Av$|El agente llega a $(7,7)$ sin colisión.|$+100$|
|$s'$ activo, para cualquier $s$ activo y $a \in A(s)$|El episodio continúa.|$-1$|
|$s$ terminal|No hay acciones aplicables, por lo que no existen tripletas.|—|



Las tres filas con recompensa son mutuamente excluyentes y cubren todas las transiciones del ambiente. Solo $Av$ puede producir $s_{\text{meta}}$, porque es la única acción que cambia la casilla del agente. La recompensa esperada de un par $(s,a)$ es:

$$
r(s,a) = -100\,P(s_{\text{col}} \mid s,a) + 100\,P(s_{\text{meta}} \mid s,a) - P(\text{activo} \mid s,a)
$$

### 5.2 Ejemplos de retorno

Con γ = 1, el retorno de un episodio es la suma de las recompensas recibidas.

|**Escenario**|**Retorno**|
|---|---|
|Ruta más corta sin obstáculos en el camino: 1 giro a la derecha y 14 avances, 15 acciones (7 hacia el este, giro, 7 hacia el sur).|$14 \cdot (-1) + 100 = \mathbf{+86}$|
|Llegar a la meta en la acción $k$.|$(k-1)(-1)+100=\mathbf{101-k}$|
|Colisión en la acción $k$.|$(k-1)(-1)-100=\mathbf{-(k+99)}$; por ejemplo, $-100$ si $k=1$.|
|Truncamiento externo después de 100 acciones sin meta ni colisión.|$100(-1)=\mathbf{-100}$; el truncamiento no añade una recompensa especial.|



### 5.3 Por qué estas recompensas

- **+100 por llegar a la meta.** Es el objetivo principal del agente y produce el mayor refuerzo individual.

- **−1 por cada acción, incluidos los giros.** Premia las rutas con menos acciones y menos giros, y evita que el agente dé vueltas indefinidamente.

- **−100 por colisión.** Es mucho mayor que el costo de los pasos de una ruta normal (de unos 15 a 30 pasos), así que el agente aprende a evitar las transiciones que históricamente han producido choques, aun sin observar directamente los obstáculos.

- **Sin recompensa especial por truncamiento.** El tope de entrenamiento es un mecanismo externo para evitar episodios indefinidos; no cambia la definición del problema ni agrega un tercer estado terminal.

## 6. Uso previsto en el aprendizaje

Se propone Q-learning tabular [2] con política $\varepsilon$-voraz sobre el estado $s=(f,c,d)$, con $\gamma = 1$ y las recompensas de la sección 5. Con $\alpha$ como tasa de aprendizaje, la actualización tras una transición $(s,a,r,s')$ es:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right].
$$

El máximo es sobre las acciones aplicables en $s'$ y vale 0 si $s'$ es terminal. La Q-tabla puede almacenarse como un arreglo de dimensiones $8 \times 8 \times 4 \times 3$ o como un diccionario. Para validar el aprendizaje se compararán la tasa de éxito, la tasa de colisión, la tasa de truncamiento y los pasos promedio hasta la meta frente a una política que elige al azar entre las acciones aplicables.

## 7. Verificación frente a las instrucciones

|**Instrucción de la entrega**|**Dónde se cumple**|
|---|---|
|Definir el conjunto de estados, con todas las<br>variables del problema.|Sección 2: estado del agente $(f,c,d)$ con tipo y rango; posiciones de obstáculos mantenidas internamente por el simulador; estados iniciales y terminales; y tamaño de 256 estados observables.|
|Escribir qué acciones puede ejecutar el agente en<br>cada estado.|Sección 3: las tres acciones y la tabla de aplicabilidad por tipo de<br>estado (3.3).|
|Especificar cómo se dan las recompensas, con<br>tripletas (s, a, sʹ).|Sección 5.1: recompensa para cada tripleta, con valores<br>numéricos y su justificación en 5.3.|
|Que las acciones correspondan al comportamiento<br>esperado del agente.|Secciones 1 y 3: coinciden con las del enunciado (rotar 90° o<br>avanzar si es posible).|
|Que la recompensa exprese incentivos positivos y<br>negativos.|Sección 5.3: incentivo positivo por llegar y negativos por chocar y por cada acción; el truncamiento externo no agrega recompensa.|



## Referencias

[1] R. S. Sutton and A. G. Barto, _Reinforcement Learning: An Introduction_ , 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[2] C. J. C. H. Watkins and P. Dayan, “Q-learning,” _Machine Learning_ , vol. 8, no. 3–4, pp. 279–292, 1992.
