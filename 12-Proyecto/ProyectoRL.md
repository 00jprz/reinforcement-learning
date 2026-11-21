Proyecto del curso

Construir un agente de RL para resolver un problema de aprendizaje específico. Para el proyecto deben ejecutar un agente de RL en el ambiente descrito a continuación, evaluar su ejecución y validar la efectividad del aprendizaje del agente entrenado sobre el ambiente.

Para esta tarea vamos a utilizar un ambiente dinámico donde el agente debe completar su tarea sin ser bloqueado por los obstáculos, los cuales se mueven aleatoriamente en cada paso. El objetivo del agente (el triangulo negro en la figura) es alcanzar el estado objetivo (el cuadrado verde en la parte inferior derecha de la figura) sin estrellarse con ninguno de los obstáculos (los círculos azules en la figura).

El ambiente consiste de un cuarto de 8x8 vacía. El agente siempre se encuentra en la posición superior izquierda, la posición (0,0) del cuarto. El estado objetivo siempre se encuentra la posición inferior derecha, en la posición (7,7). Sin embargo, dentro del cuarto aparecen unos obstáculos de forma aleatoria (5 en total) los cuales deben ser evitados por el agente. El agente debe evitar los obstáculos en el ambiente, de estrellarse con alguno de ellos, el episodio se da por terminado.

En un pasó de la ejecución, los obstáculos cambian de posición a (posiblemente) una nueva posición, al mismo tiempo que el agente toma la decisión sobre la acción a tomar. El agente tiene dos grados de libertad para su movimiento, rotar sobre su eje 90º a la izquierda o derecha, o moverse a la casilla inmediatamente frente a él (si esto es posible).

Su misión es definir el problema (i.e., acciones, estados y sistema de recompensas) para permitir al agente alcanzar el objetivo. 
