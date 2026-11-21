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
    
    def get_states(self):
        return self.grid

    def get_current_state(self):
        return self.state
    
    def state_value(self, state):
        return self.grid[state[0]][state[1]]
    
    def get_possible_actions(self, state):
        i, j = state
        actions = ['up', 'right', 'down', 'left']
        if self.grid[i][j] == 1 or self.grid[i][j] == -1:
            actions = ['exit']
        elif self.grid[i][j] == None:
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


        # Definición de las librerías a utilizar

import random

#PolicyIteraction
class PolicyIteration():
    #1
    def __init__(self, mdp, discount=0.9, iterations=64):
        ### BEGIN SOLUTION
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.last_values = {}
        self.values = {} # {(i,j): 0 for i in range(self.mdp.nrows) for j in range(self.mdp.ncols)}
        self.policy = {}
        self.last_policy = []
        for i in range(mdp.nrows):
            for j in range(mdp.ncols):
                state = (i,j)
                actions = self.mdp.get_possible_actions(state)
                if actions == []:
                    self.policy[state] = None
                else:
                    self.policy[state] = random.choice(actions)
        #END SOLUTION
    
    def get_policy(self, state):
        ### BEGIN SOLUTION
        return self.policy[state]
        ### END SOLUTION
    
    def get_value(self, state):
        ### BEGIN SOLUTION
        if self.values.get(state) == None:
           self.values[state] = 0
        return self.values[state]
        ### END SOLUTION
    
    def get_action(self, state):
        ### BEGIN SOLUTION
        return self.policy[state]
        ### END SOLUTION

    def compute_new_action(self, state):
        ### BEGIN SOLUTION
        best_action = self.policy[state]
        best_value = self.get_value(state)
        max_value, max_action = self.compute_new_value(state)
        if max_value > best_value:
            best_value = max_value
            best_action = max_action
        #self.policy[state] = best_action
        return best_action
        ### END SOLUTION

    def value_evaluation(self, state, action):
        ### BEGIN SOLUTION
        if not self.mdp.is_terminal(state):
            probs, rewards, states = self.mdp.get_possible_states(state, action)
            val = 0
            for i in range(len(probs)):
                val += probs[i]*(rewards[i] + self.discount*self.get_value(states[i]))
            return val
        else :
            return self.mdp.state_value(state)
        ### END SOLUTION

    def compute_new_value(self, state):
        ### BEGIN SOLUTION
        if not self.mdp.is_terminal(state):
            max_val = float('-inf')
            action = ""
            for a in self.mdp.get_possible_actions(state):
                probs, rewards, states = self.mdp.get_possible_states(state, a)
                val = 0
                for i in range(len(probs)):
                    val += probs[i]*(rewards[i] + self.discount*self.get_value(states[i]))
                if val >= max_val:
                    max_val = val
                    action = a
            return max_val, action
        else :
            return self.mdp.state_value(state), self.policy[state]
        ### END SOLUTION
    
    def policy_evaluation(self, new_policy):
        ### BEGIN SOLUTION
        improvement = False
        for i in range(self.mdp.nrows):
            for j in range(self.mdp.ncols):
                if self.policy[(i,j)] != new_policy[(i,j)]:
                    improvement = True
                    return improvement
        return improvement
        ### END SOLUTION
    
    def policy_iteration(self):
        ### BEGIN SOLUTION
        new_values = {}
        for i in range(self.mdp.nrows):
            for j in range(self.mdp.ncols):
                state = (i,j)
                if self.mdp.grid[i][j] != None:
                    new_values[state] = self.get_value(state)
                    new_values[state] = self.value_evaluation(state, self.policy[state])
        self.values = new_values
        # Mejora
        new_policy = {}
        for i in range(self.mdp.nrows):
            for j in range(self.mdp.ncols):
                state = (i,j)
                new_policy[(i,j)] = self.policy[(i,j)]
                if self.mdp.grid[i][j] != None:
                    if i==3 and j==1:
                        new_policy[(i,j)] = self.compute_new_action(state)
                    new_policy[(i,j)] = self.compute_new_action(state)
        if self.policy_evaluation(new_policy):
            #print("Convergencia en la iteracion:",it)
            self.policy = new_policy
            return False, self.policy
        return True, self.policy
        ### END SOLUTION
    
    def run_policy_iteration(self):
        done = False
        i = 1
        policy = {}
        while not done and i <= self.iterations:
            done, policy = self.policy_iteration()
            i += 1
        if done:
            print(f"La política converge en {i} iteraciones")
            print(policy)
        return self.values, self.policy  



#Ambiente general de prueba para todos los casos siguientes

board = [[' ', ' ', ' ', '+1'],
      [' ', '#', ' ', '-1'],
      [' ', ' ', '#', ' '],
      ['S', ' ', ' ', ' ']]

gridworld = Gridworld(board, (4,4))


#Pruebas de policy_evaluation

a = PolicyIteration(gridworld, 0.8, 50)

a.run_policy_iteration()