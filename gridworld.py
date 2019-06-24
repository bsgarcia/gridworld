import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import time


class GUI:

    def __init__(self, nrow, ncol, path, update_freq=0.005):
        self.gui_grid = np.zeros((nrow, ncol))
        self.im = self.get_im(path, self.gui_grid)
        self.update_freq = update_freq
        plt.show(block=False)

    def update_gui(self, position, t, fails, successes):
        # If a window is open
        if plt.get_fignums():
            # set player's position
            self.gui_grid[position] += 2
            self.im.set_array(self.gui_grid)
            # reinit the grid values
            self.gui_grid[position] -= 2
            plt.title(f't={t}, fails={fails}, successes={successes}')
            plt.pause(self.update_freq)
        # Else exit the program
        else:
            exit('You closed the window')

    @staticmethod
    def get_im(path, gui_grid):
        for i in path:
            gui_grid[i] += 1
        im = plt.imshow(gui_grid, cmap="Pastel2", vmax=2, vmin=0)
        return im


class Environment:

    def __init__(self, ncol=5, nrow=5, t_max=1000):

        self.t_max = t_max

        self.ag = Agent(
            nrow=nrow,
            ncol=ncol,
        )
        
        # For gui element
        self.fig = None
        self.im = None

        # --------------------------- init grid -----------------------#

        self.punishment = -1
        self.grid = np.ones((nrow, ncol), dtype=object) * self.punishment

        # compute path to the goal based on matrix size
        path = self.compute_path(nrow)

        # The rewards are more and more valuable along the path
        self.rewards = np.arange(len(path))

        for pos, reward in zip(sorted(path), self.rewards):
            self.grid[pos] = reward
        # ----------------------------------------------------------- #
        # init gui
        self.gui = GUI(nrow=nrow, ncol=ncol, path=path)

    def run(self):

        fails = 0
        successes = 0

        for t in range(self.t_max):

            self.gui.update_gui(self.ag.position, t, fails, successes)

            action, old_pos, new_pos = self.ag.move()
            reward = self.get_reward(position=new_pos)
            self.ag.learn(position=old_pos, action=action, reward=reward)

            fails += reward == self.punishment
            successes += reward == max(self.rewards)
            restart = (reward == self.punishment) or (reward == max(self.rewards))

            self.gui.update_gui(
                    self.ag.position, t, fails, successes)

            if restart:
                self.ag.restart()

    def get_reward(self, position):
        """récupère la récompense de l'agent"""
        return self.grid[position]

    @staticmethod
    def compute_path(nrow):
        path = []
        for i in range(nrow):
            if i == 0:
                path.append((i, i))
            if i + 1 != nrow:
                path.append((i + 1, i))
                path.append((i + 1, i + 1))
        return path


class Agent:

    def __init__(self, ncol, nrow, epsilon=0.3, alpha=0.5, q=0.5, initial_position=(0, 0)):

        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # droite, gauche, bas, haut)

        # array avec les q_values de chaque actions à chaque état (4 actions et nrow*ncol états)
        self.all_positions = list(it.product(range(ncol), range(nrow))) #on crée une liste avec toute les positions, it.product permet de créer toute les combinaisons entre les coordonnées des vecteurs
        self.q = {
            (x, y): np.ones(len(self.actions)) * q for x, y in self.all_positions
        } #on crée un dico d'array avec les qvalues et l'emplacement en clé

        self.position = initial_position
        self.initial_position = initial_position

        self.epsilon = epsilon
        self.alpha = alpha
    
    def restart(self):
        # slice ([:]) the initial_position tuple otherwise self.position
        # and self.initial_position are considered the same variable
        self.position = self.initial_position[:]

    def move(self):
        """choix de l'action de l'agent, en fonction de son coef epsilon"""

        action = self.get_available_actions()

        old_pos = self.position

        selected_action = np.random.choice(action, p=self.softmax(old_pos, action))

        self.position = self.increment_position(int(selected_action))

        # Return action, old pos, new pos
        return selected_action, old_pos, self.position

    def increment_position(self, action):
        x1, y1 = self.position
        x2, y2 = self.actions[action]
        return x1 + x2, y1 + y2

    def get_available_actions(self):
        x1, y1 = self.position
        return np.asarray(
            [i for i, (x2, y2) in enumerate(self.actions)
             if (x1 + x2, y1 + y2) in self.all_positions]
        )

    def learn(self, position, action, reward):
        """acutalise la q_values"""
        self.q[position][action] += self.alpha * (reward - self.q[position][action])

    def softmax(self, position, action):
        return np.exp(self.q[position][action]) / np.sum(np.exp(self.q[position][action]))


def main():
    env = Environment()
    env.run()
    

if __name__ == '__main__':
    main()
