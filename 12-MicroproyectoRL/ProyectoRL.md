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

Las tres acciones (G_izq, G_der, Av) están definidas, con el efecto sobre fila, columna y orientación. A(s) cubre todos los casos: las tres si la casilla de enfrente existe; solo giros en el borde; ninguna en terminales (s_meta, s_col, s_lim). Avanzar con obstáculo al frente sí aplica. Coinciden con el comportamiento del enunciado. Bien que el giro cuente como paso y mueva obstáculos.  

### Definición de los estados

1. Los estados están definidos completamente
2. Los estados están definidos apropiadamente con respecto al problema de la aplicación. El estado contiene todas las características para que los agentes aprendan
3. Los estados cubren todas las características del ambiente

El estado del ambiente (f,c,d,O,t) incluye posición, orientación y los cinco obstáculos, con tipo y rango. Eso describe el cuarto. El problema es la observación para la Q-tabla: o(s)=(f,c,d,L) con 12 bits (casillas a distancia de Manhattan ≤2). Eso es un mapa local de obstáculos. El agente no ve si hay un círculo cerca; se entera al chocar. Con L el problema deja de ser el del curso. Para aprender basta (f,c,d); O se queda en el simulador. Además: el enunciado no pide el contador t ni un tercer terminal s_lim; si usan un tope, que sea un corte externo, no parte de s. Por los sensores, los estados no están definidos de forma apropiada para esta aplicación.  

### Recompensas

1. Las recompensas permiten el aprendizaje del agente o cubren todos los casos
2. Las recompensas están correctamente definidas

Definen R(s,a,s′) por triplas: +100 meta, −100 colisión, −1 si el episodio sigue, −50 si se acaba el límite. Hay incentivo positivo y negativo y cubren las transiciones legales. Ojo en el entrenamiento: un choque temprano (≈−100)sale mejor que agotar el límite (−149); el agente podría preferir estrellarse. El enunciado no pide multa por tiempo; con −1 por paso ya se cobra la tardanza. Si mantienen s_lim, igualen o quiten esa penalización extra para que chocar siga siendo lo peor.  






