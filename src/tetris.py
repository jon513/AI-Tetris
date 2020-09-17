import pygame
import random
import sys
import numpy as np

import copy

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)
END_OF_GAME = "gameover"
EXPECTIMAX = 'expectimax'
ALPHA_BETA = 'alpha_beta'
REFLEX = 'reflex'
Q_LEARNING = 'q_learning'
USAGE_MSG = '\nUsage: <agent> <**kwargs>\n'
INVALID_HEURISTIC_MSG = '\nNot a valid Heuristic.'

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    """class repesenting the diffrent pices in Tetris using a 4x4 matrix"""
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        # randomly pick a type and a color.
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0
        self.all_rotations = np.arange(len(self.figures[self.type]))

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])

    def rotate_to_index(self, index):
        """rotate to an index"""
        # TODO: check that index is valid
        self.rotation = index


class Tetris:
    level = 2
    score = 0
    state = "start"
    field = []  # field of the game that contains zeros where it is empty, and the colors where there are figures
    height = 0
    width = 0
    x = 100
    y = 60
    zoom = 20
    figure = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        """adding new figure to the board"""
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for row in range(4):
            for col in range(4):
                if row * 4 + col in self.figure.image():
                    if row + self.figure.y > self.height - 1 or \
                            col + self.figure.x > self.width - 1 or \
                            col + self.figure.x < 0 or \
                            self.field[row + self.figure.y][col + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = END_OF_GAME

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation


class Game:

    def __init__(self, agent, rows=20, cols=10, environment=None):
        """:param agent: A function that returns a list with a single Event object"""
        # Initialize the game engine
        pygame.init()

        self.screen = pygame.display.set_mode(size)

        pygame.display.set_caption("Tetris")

        # Loop until the user clicks the close button.
        self.done = False
        self.clock = pygame.time.Clock()
        self.fps = 25
        self.game = Tetris(rows, cols)
        self.agent = agent

    def run(self):
        counter = 0
        pressing_down = False

        while not self.done:
            if self.game.figure is None:
                self.game.new_figure()
            counter += 1
            if counter > 100000:
                counter = 0

            if counter % (self.fps // self.game.level // 2) == 0 or pressing_down:
                if self.game.state == "start":
                    self.game.go_down()

            for event in list(pygame.event.get()) + self.agent(self.game):
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.game.rotate()
                    if event.key == pygame.K_DOWN:
                        pressing_down = True
                    if event.key == pygame.K_LEFT:
                        self.game.go_side(-1)
                    if event.key == pygame.K_RIGHT:
                        self.game.go_side(1)
                    if event.key == pygame.K_SPACE:
                        self.game.go_space()
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        pressing_down = False

            self.screen.fill(WHITE)
            for i in range(self.game.height):
                for j in range(self.game.width):
                    pygame.draw.rect(self.screen, GRAY,
                                     [self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i,
                                      self.game.zoom, self.game.zoom], 1)
                    if self.game.field[i][j] > 0:
                        pygame.draw.rect(self.screen, colors[self.game.field[i][j]],
                                         [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1,
                                          self.game.zoom - 2, self.game.zoom - 1])

            if self.game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in self.game.figure.image():
                            pygame.draw.rect(self.screen, colors[self.game.figure.color],
                                             [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                              self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                              self.game.zoom - 2, self.game.zoom - 2])

            font = pygame.font.SysFont('Calibri', 25, True, False)
            font1 = pygame.font.SysFont('Calibri', 65, True, False)
            text = font.render("Score: " + str(self.game.score), True, BLACK)
            text_game_over = font1.render("Game Over :( ", True, (255, 0, 0))

            self.screen.blit(text, [0, 0])
            if self.game.state == END_OF_GAME:
                self.screen.blit(text_game_over, [10, 200])
                self.done = True
                print('Game Over! Score = %d' % self.get_score())

            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()

    def game_over(self):
        """

        :return: true if games over
        """
        if self.game.state == END_OF_GAME:
            return True
        return False

    def get_score(self):
        """
        :return: games score
        """
        return self.game.score


def user_input(dummy_arg=None):
    """Dummy function to disallow AI input"""
    return []


def get_heuristic_func(sys_arg):
    """Takes the heuristic string and returns the relevant function"""
    return None  # TODO Fill in relevant if/else function


def print_agents(agents):
    print("Available Agents:")
    for agent in agents:
        print('\t- %s' % agent)


def print_heuristics(heuristics):
    print('Available Heuristics:')
    for heuristic in heuristics:
        print('\t- %s' % heuristic)


if __name__ == "__main__":
    import tetris_ai
    import argparse

    parser = argparse.ArgumentParser(description="get an agent name, depth and board size ")
    parser.add_argument('-a', dest='agent', type=str)
    parser.add_argument('-d', dest='depth', type=int, default=1)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.2)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.75)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.05)
    parser.add_argument('--num_training', dest='num_training', type=int, default=10)
    parser.add_argument('--num_testing', dest='num_testing', type=int, default=5)
    parser.add_argument('--gui', dest='gui', type=bool, default=True)
    parser.add_argument('-s', nargs=2, type=int, dest='size', default=[20, 10])
    args = parser.parse_args()
    args_dict = vars(args)  # convert argparse namespace to a dictionary

    if len(sys.argv) == 1:  # No Agent -- Keyboard input
        agent = user_input
    else:
        agents = [EXPECTIMAX, ALPHA_BETA, REFLEX, Q_LEARNING]
        if args.agent not in agents:
            print(USAGE_MSG)
            print_agents(agents)
            sys.exit(1)
        elif not args.gui:
            from qlearningAgents import ApproximateQAgent
            agent = ApproximateQAgent(
                alpha=args_dict['alpha'],
                gamma=args_dict['gamma'],
                epsilon=args_dict['epsilon'],
                num_training=args_dict['num_training'],
                num_testing=args_dict['num_testing'],
                gui=False
            )
            agent.run_testing_rounds()
            sys.exit(0)
        else:
            agent = tetris_ai.get_ai_agent(args.agent, args_dict)
    game = Game(agent, rows=args.size[0], cols=args.size[1])
    game.run()
