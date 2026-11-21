#Ambiente de gridworld

import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Definición de la clase principal
class Gridworld:
    def __init__(self, board, dimensions):
        self.nrows, self.ncols = dimensions
        self.grid = [[0 for _ in range(self.ncols)] for _ in range(self.nrows)]
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == "#":
                    self.grid[i][j] = None
                elif board[i][j] == "S":
                    self.initial_state = (i,j)
                elif board[i][j] != " ":
                    self.grid[i][j] = int(board[i][j])
        self.state = self.initial_state
    
    def get_current_state(self):
        return self.state
    
    def state_value(self, state):
        return self.grid[state[0]][state[1]]
    
    def get_possible_actions(self, state):
        i, j = state
        actions = ['up', 'right', 'down', 'left']
        if self.grid[i][j] == None:
            actions = []
        return actions
    
    def get_action_index(self, action):
        actions = ['up', 'right', 'down', 'left']
        index = 0
        for a in actions:
            if action == a:
                return index
            index += 1

    def get_possible_states(self, state, action):
        actions = self.get_possible_actions(state)
        action_index = self.get_action_index(action)
        probability = [0.1, 0.1, 0.8]
        rewards=[]
        states=[]
        for i in [-1, 1, 0]:
            action = actions[(action_index+i)%len(actions)]
            reward, new_state = self.simulate_action(state, action) 
            rewards.append(reward)
            states.append(new_state)
        return probability, rewards, states
        
    def simulate_action(self, state, action):
        i,j = state
        if action == 'up' and i > 0 and self.grid[i-1][j] != None:
            i -= 1
        elif action == 'down' and i < self.nrows-1 and self.grid[i+1][j] != None:
            i += 1
        elif action == 'left' and j > 0 and self.grid[i][j-1] != None:
            j -= 1
        elif action == 'right' and j < self.ncols-1 and self.grid[i][j+1] != None:
            j += 1
        new_state = (i,j)
        return self.grid[i][j], new_state
        
    def do_action(self, action):
        i, j = self.state
        if action == 'up' and i > 0 and self.grid[i-1][j] != None:
            i -= 1
        elif action == 'down' and i < self.nrows-1 and self.grid[i+1][j] != None:
            i += 1
        elif action == 'left' and j > 0 and self.grid[i][j-1] != None:
            j -= 1
        elif action == 'right' and j < self.ncols-1 and self.grid[i][j+1] != None:
            j += 1
        self.state = (i,j)
        return self.grid[i][j], self.state

    def reset(self):
        self.state = self.initial_state
    
    def is_terminal(self, state=None):
        if state == None:
            state = self.state
        return self.grid[state[0]][state[1]] == 1 or self.grid[state[0]][state[1]] == -1
    
    #Función de soporte para graficar el ambiente
    def plot(self):
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111, aspect='equal')
        
        # Lineas
        for i in range(0, len(self.grid)+1):
            ax1.axhline(i , linewidth=2, color="#2D2D33")
            ax1.axvline(i , linewidth=2, color="#2D2D33")
        
        # Amarillo - estado inicial
        (i,j)  = self.initial_state
        ax1.add_patch(patches.Rectangle((j, self.nrows - i -1), 1, 1, facecolor = "#F6D924"))
        for j in range(len(self.grid[0])):
            for i in range(len(self.grid)):
                if self.grid[i][j] == 1: # verde
                    ax1.add_patch(patches.Rectangle((j,self.nrows - i -1), 1, 1, facecolor = "#68FF33"))
                if self.grid[i][j] == None: # gris
                    ax1.add_patch(patches.Rectangle((j,self.nrows - i -1), 1, 1, facecolor = "#6c7780"))
                if self.grid[i][j] == -1: # rojo
                    ax1.add_patch(patches.Rectangle((j,self.nrows - i -1), 1, 1, facecolor = "#cc0000"))
        plt.scatter(self.state[1] + 0.5, self.nrows - self.state[0] - 1 +0.5, s = 100, color = "black", marker = "o", facecolor = "blue", edgecolors = "blue", zorder = 10)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == None:
                    ax1.text(self.ncols-j-1, self.nrows-i-1, "", ha='center', va='center')
                else:
                    ax1.text(j+0.5, self.nrows-i-1+0.5, self.grid[i][j], ha='center', va='center')
        plt.axis("off")
        plt.show()
    

class ValueIteration():    
    def __init__(self, mdp, discount = 0.9, iterations = 64):
    ### BEGIN SOLUTION
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {} # {(i,j): 0 for i in range(self.mdp.nrows) for j in range(self.mdp.ncols)}
    ### END SOLUTION
    
    def get_value(self, state):
        ### BEGIN SOLUTION
        if self.values.get(state) == None:
           self.values[state] = 0
        return self.values[state]
        ### END SOLUTION

    def compute_new_value(self, state):
        ### BEGIN SOLUTION
        if not self.mdp.is_terminal(state):
            max_val = float('-inf')
            for action in self.mdp.get_possible_actions(state):
                probs, rewards, states = self.mdp.get_possible_states(state, action)
                val = 0
                for i in range(len(probs)):
                    val += probs[i]*(rewards[i] + self.discount*self.get_value(states[i]))
                if val > max_val:
                    max_val = val
            return max_val
        else :
            return self.mdp.state_value(state)
        ### END SOLUTION

    def compute_action_from_values(self, state):
        ### BEGIN SOLUTION
        best_action = None
        best_value = float('-inf')
        for action in self.mdp.get_possible_actions(state):
            new_value = self.compute_new_from_values(state, action)
            if new_value > best_value:
                best_value = new_value
                best_action = action
        return best_action
        ### END SOLUTION

    def get_action(self, state):
        ### BEGIN SOLUTION
        return self.compute_action_from_values(state)
        ### END SOLUTION
        
    def run_value_iteration(self):
        ### BEGIN SOLUTION
        for _ in range(self.iterations):
            new_values = self.values.copy()
            for i in range(len(self.mdp.get_states())):
                for j in range(len(self.mdp.get_states()[0])):
                    values = []
                    state = (i,j)
                    for action in self.mdp.get_possible_actions(state):
                        values.append(self.compute_new_value_from_values(state, action))
                    if values:
                        new_values[state[0]][state[1]] = max(values)
            self.values = new_values
        return self.values
        ### END SOLUTION

#Ambiente general de prueba para todos los casos siguientes

board = [[' ', ' ', ' ', '+1'],
      [' ', '#', ' ', '-1'],
      [' ', ' ', '#', ' '],
      ['S', ' ', ' ', ' ']]

gridworld = Gridworld(board, (4,4))

#Pruebas de compute_new_value

a = ValueIteration(gridworld, 0.8, 50)

### BEGIN TESTS
try:
    a.compute_new_value
except:
    print("La función compute_new_value no está definida")

print(a.compute_new_value((0,3)))
#assert a.get_value((4,1)) == 1, f"la función get_value no esta retornando el valor guardado en el mapa responde {a.get_value((0,4))} y no 1"
#assert a.get_value((0,0)) == 0, f"la función get_value no devuelve el valor de 0 para los estados no visitados, devuelve {a.get_value((0,0))}"

### END TEST