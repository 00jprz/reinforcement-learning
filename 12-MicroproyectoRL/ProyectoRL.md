# Proyecto del curso

Construir un agente de RL para resolver un problema de aprendizaje específico. Para el proyecto deben ejecutar un agente de RL en el ambiente descrito a continuación, evaluar su ejecución y validar la efectividad del aprendizaje del agente entrenado sobre el ambiente.

Para esta tarea vamos a utilizar un ambiente dinámico donde el agente debe completar su tarea sin ser bloqueado por los obstáculos, los cuales se mueven aleatoriamente en cada paso. El objetivo del agente (el triangulo negro en la figura) es alcanzar el estado objetivo (el cuadrado verde en la parte inferior derecha de la figura) sin estrellarse con ninguno de los obstáculos (los círculos azules en la figura).

El ambiente consiste de un cuarto de 8x8 vacía. El agente siempre se encuentra en la posición superior izquierda, la posición (0,0) del cuarto. El estado objetivo siempre se encuentra la posición inferior derecha, en la posición (7,7). Sin embargo, dentro del cuarto aparecen unos obstáculos de forma aleatoria (5 en total) los cuales deben ser evitados por el agente. El agente debe evitar los obstáculos en el ambiente, de estrellarse con alguno de ellos, el episodio se da por terminado.

En un pasó de la ejecución, los obstáculos cambian de posición a (posiblemente) una nueva posición, al mismo tiempo que el agente toma la decisión sobre la acción a tomar. El agente tiene dos grados de libertad para su movimiento, rotar sobre su eje 90º a la izquierda o derecha, o moverse a la casilla inmediatamente frente a él (si esto es posible).

Su misión es definir el problema (i.e., acciones, estados y sistema de recompensas) para permitir al agente alcanzar el objetivo. 


## Resultados de la rubrica:

### Definición de las acciones del agente

1. Las acciones están debidamente definidas
2. La definición de las acciones esta completa 
3. El agente define todas las acciones que permiten el comportamiento de la aplicación

retroalimentación: Las tres acciones (G_izq, G_der, Av) están definidas, con el efecto sobre fila, columna y orientación. A(s) cubre todos los casos: las tres si la casilla de enfrente existe; solo giros en el borde; ninguna en terminales (s_meta, s_col, s_lim). Avanzar con obstáculo al frente sí aplica. Coinciden con el comportamiento del enunciado. Bien que el giro cuente como paso y mueva obstáculos.  

### Definición de los estados

1. Los estados están definidos completamente
2. Los estados están definidos apropiadamente con respecto al problema de la aplicación. El estado contiene todas las características para que los agentes aprendan
3. Los estados cubren todas las características del ambiente

retroalimentación: El estado del ambiente (f,c,d,O,t) incluye posición, orientación y los cinco obstáculos, con tipo y rango. Eso describe el cuarto. El problema es la observación para la Q-tabla: o(s)=(f,c,d,L) con 12 bits (casillas a distancia de Manhattan ≤2). Eso es un mapa local de obstáculos. El agente no ve si hay un círculo cerca; se entera al chocar. Con L el problema deja de ser el del curso. Para aprender basta (f,c,d); O se queda en el simulador. Además: el enunciado no pide el contador t ni un tercer terminal s_lim; si usan un tope, que sea un corte externo, no parte de s. Por los sensores, los estados no están definidos de forma apropiada para esta aplicación.  

### Recompensas

1. Las recompensas permiten el aprendizaje del agente o cubren todos los casos
2. Las recompensas están correctamente definidas

retroalimentación: Definen R(s,a,s′) por triplas: +100 meta, −100 colisión, −1 si el episodio sigue, −50 si se acaba el límite. Hay incentivo positivo y negativo y cubren las transiciones legales. Ojo en el entrenamiento: un choque temprano (≈−100)sale mejor que agotar el límite (−149); el agente podría preferir estrellarse. El enunciado no pide multa por tiempo; con −1 por paso ya se cobra la tardanza. Si mantienen s_lim, igualen o quiten esa penalización extra para que chocar siga siendo lo peor.  



## Dudas resueltas 

#### El agente puede salirse del cuarto en el proyecto? 
Respuestas: No. El cuarto es 8×8 y no hay casillas afuera. Si está en el borde y mira hacia afuera, no puede avanzar: esa acción no existe en ese estado. Solo puede girar. No se cae del mapa.

#### El agente choca cuando se mueve sin saber q está enfrente (cuando por azar el obstáculo cae en la casilla donde se movió)? Quiero entender si el choque se da antes o después del salto
Respuestas: Sí. El agente no ve obstáculos. En cada paso, al mismo tiempo: él actúa (gira o avanza) y los 5 obstáculos se mueven al azar. Después se mira el resultado. Hay choque si: 
    1. él y un obstáculo terminan en la misma casilla, o
    2. se cruzan (él va de A a B y el obstáculo de B a A en el mismo paso). También puede chocarse sin avanzar: si solo gira y un obstáculo cae en su casilla.

#### los obstáculos pueden caer en una misma casilla? 
Respuesta: No es “antes” ni “el agente salta y luego las bolitas”. Es a la vez, y el choque se decide al terminar el paso, cuando ya se sabe dónde quedó cada uno. Los obstáculos no pueden caer uno sobre otro. Tampoco pueden caer en la meta (7, 7).

#### El inicio es siempre como está en la foto de la casilla? O solo se sabe que el agente está ubicado como está en la foto pero los obstáculos pueden caer en cualquier casilla incluido donde están el agente y la meta?

Respuesta: No. La foto es un ejemplo. Lo que sí es fijo siempre:
    1. agente en (0, 0) (esquina superior izquierda),
    2. meta en (7, 7) (esquina inferior derecha).
Los 5 obstáculos empiezan al azar (no tienen que coincidir con la foto). Pueden, en un paso, quedarse en la misma casilla: el azar no obliga a moverse.

#### Alcance del movimiento: en un paso, ¿cada obstáculo se mueve (a) a una casilla vecina, o (b) a cualquier casilla libre del tablero? Si es (a), ¿las vecinas son solo arriba/abajo/izquierda/derecha, o también las diagonales?
Respuesta: Maneja el movimiento de los obstáculos de manera aleatoria eso te ayudará a que no debas trabajar con probabilidades

#### Probabilidades: ¿todas las opciones (incluida quedarse quieto) tienen la misma probabilidad, o podemos definirlas nosotras?
Respuesta: Justamente para evitar jugar con las probabilidades es preferible que uses el movimiento aleatorio de los obstáculos

#### Movimiento bloqueado: si el destino elegido para un obstáculo no es válido (pared, otro obstáculo o la meta), ¿el obstáculo se queda quieto o se elige otro destino?
Respuesta: El movimiento se moverá aleatoriamente dentro de la grilla, si tu defines correctamente tu ambiente los obstáculos no se moverán por fuera de la grilla, y como consejo invalida que se muevan sobre la posición de salida y de meta.

#### Inicio: ¿un obstáculo puede empezar en (0,0) o en una casilla vecina al agente? ¿Hacia dónde mira el agente al inicio: una dirección fija (¿cuál?) o al azar?
Respuesta: No debería poder empezar sobre la posición 0,0 porque es la de partida, su posición es aleatoria cumpliendo lo que en el pto 3 te contesté.

#### Límite de pasos: ¿hay un número máximo de pasos por episodio? Si se alcanza, ¿cuenta como fracaso, o simplemente se reinicia sin éxito ni choque?
Respuesta: No hay un límite de pasos.

#### Post en Foros: Raúl sugirió en el foro que el agente se mueve primero y luego los obstáculos, y que dos obstáculos sí podrían coincidir, aclarando que eran decisiones de modelado. ¿Lo que tú indicas son reglas obligatorias, o también son decisiones que podemos tomar si las justificamos?
Respuesta: El que coincidan o no los obstáculos en alguna ocasión puede darse pero como es aleatoria no sucederá siempre.

#### Uso de librerías: ¿podemos usar el ambiente DynamicObstacles de MiniGrid como referencia o como código base, o el ambiente debe estar implementado completamente por nosotras?
Respuesta: Si como te lo expliqué mientras no incidan en el comportamiento del agente si puedes usarla.



