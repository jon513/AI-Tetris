import numpy as np

import pygame
import random

END_OF_GAME = 'gameover'
PLAYING_GAME = 'playing'


class Event:
    type = None
    key = None

    def __init__(self, type, key):
        self.type = type
        self.key = key


class GameState:
    """
    This class is used to replace the Tetris class (which is the game engine) when simulating the game in order to
    gain efficiency.
    """

    numpy_figures = [  # a map of figures the correspond to the figure map in the Tetris class.
        [(np.array([0, 1, 2, 3]), np.array([4, 4, 4, 4])), (np.array([0, 0, 0, 0]), np.array([3, 4, 5, 6]))],

        [(np.array([0, 0, 1, 2]), np.array([4, 5, 4, 4])), (np.array([0, 1, 1, 1]), np.array([3, 3, 4, 5])),
         (np.array([0, 1, 2, 2]), np.array([4, 4, 4, 3])), (np.array([0, 0, 0, 1]), np.array([3, 4, 5, 5]))],

        [(np.array([0, 0, 1, 2]), np.array([4, 5, 5, 5])), (np.array([0, 0, 0, 1]), np.array([4, 5, 6, 4])),
         (np.array([0, 1, 2, 2]), np.array([5, 5, 5, 6])), (np.array([0, 1, 1, 1]), np.array([6, 4, 5, 6]))],

        [(np.array([0, 1, 1, 1]), np.array([4, 3, 4, 5])), (np.array([0, 1, 2, 1]), np.array([4, 4, 4, 3])),
         (np.array([0, 0, 0, 1]), np.array([3, 4, 5, 4])), (np.array([0, 1, 2, 1]), np.array([4, 4, 4, 5]))],

        [(np.array([0, 0, 1, 1]), np.array([4, 5, 4, 5]))]
    ]

    def __init__(self, game_field, figure_type):
        self.field = np.array(game_field, dtype=np.int8)
        self.field[np.where(self.field != 0)] = 1  # ignore the colors of the Tetris object.
        self.rotation = 0
        self.score = 0
        self.figure_type = figure_type

        # get a unique copy of a figure from the figure map. When moving the piece along the axis in a certain
        # game_state it does not change for the other states:
        self.figure_index = np.copy([self.numpy_figures[figure_type][self.rotation][0],
                                     self.numpy_figures[figure_type][self.rotation][1]])
        self.state = PLAYING_GAME
        if np.any(self.field[tuple(self.figure_index)] == 1):
            self.state = END_OF_GAME

    def new_figure(self, figure_type=None):
        """Adds a new figure to the board. Will use a random figure if no type is supplied"""
        if figure_type is None:
            self.figure_type = random.randint(0, len(self.numpy_figures) - 1)
        else:
            self.figure_type = figure_type
        self.rotation = 0
        self.figure_index = np.copy([self.numpy_figures[self.figure_type][self.rotation][0],
                                     self.numpy_figures[self.figure_type][self.rotation][1]])

    def break_lines(self):
        """
        Similar to the break_lines method of Tetris
        """
        full_lines = np.where(self.field.all(axis=1))[0]
        if len(full_lines > 0):
            self.field = np.delete(self.field, full_lines, 0)
            new_lines = np.zeros((len(full_lines), self.field.shape[1]), dtype=np.int8)
            self.field = np.vstack((new_lines, self.field))
            self.field.dtype = np.int8
        self.score += len(full_lines) ** 2

    def generate_agent_successors_list(self):
        """
        return a list containing every game_state the can be obtained from moving the piece as follows. first, the
        rotation is set, then the x coordinate. The falling piece is moved only right steps or left steps, excluding any
        combination of the two.
        """
        output = []
        for rotation in range(len(self.numpy_figures[self.figure_type])):

            for i in range(-4, len(self.field[0])):
                successor = GameState(self.field, self.figure_type)

                #  apply rotations
                for j in range(rotation):
                    successor.rotate()

                #  apply movement on the x axis
                if successor.push_piece_x(i):
                    successor.push_piece_y()
                    output.append(successor)
        return output

    def generate_agent_successors_dict(self):
        """
        Same as generate_agent_successors_list but also map the generated game_state objects to the actions that
        generate them. The actions are of the form of Event objects, following the tetris_ai api.
        """
        output = {}
        for rotation in range(len(self.numpy_figures[self.figure_type])):

            for i in range(-5, len(self.field[0])):
                successor = GameState(self.field, self.figure_type)
                event_sequence = []

                #  apply rotations
                for j in range(rotation):
                    successor.rotate()
                    event_sequence.append(Event(pygame.KEYDOWN, pygame.K_UP))

                #  apply movement on the x axis
                if i > 0:
                    for k in range(i):
                        event_sequence.append(Event(pygame.KEYDOWN, pygame.K_RIGHT))
                elif i < 0:
                    for m in range(-i):
                        event_sequence.append(Event(pygame.KEYDOWN, pygame.K_LEFT))
                if successor.push_piece_x(i):
                    successor.push_piece_y()
                    output[successor] = event_sequence

        return output

    def generate_opponent_successors_list(self):
        """
        Return a list of all the state obtained by adding a new falling piece. There are always exactly 5 such states
        since there are only 5 pieces in the game and the starting rotation of each of them is always the same.
        """
        successors = []
        for figure_number in range(len(self.numpy_figures)):
            successor = GameState(self.field, figure_number)
            successors.append(successor)
        return successors

    def push_piece_x(self, dx):
        """
        move the falling piece dx step, where the direction is set by the sign - negative means move left, positive
        right.
        """
        self.figure_index[1] += dx
        if np.any(self.figure_index[1] < 0) or np.any(self.figure_index[1] >= len(self.field[0])) or np.any(
                self.field[tuple(self.figure_index)] == 1):
            self.figure_index[1] -= dx
            return False
        return True

    def push_piece_y(self):
        """
        move the falling piece down until it touches a taken cell or the border of the field. Then, fix the piece by
        adding its final location to the field (changing adequate 0's to 1's)
        """
        while np.all(self.figure_index[0] + 1 < len(self.field)) and \
                np.all(self.field[(self.figure_index[0] + 1, self.figure_index[1])] == 0):
            self.figure_index[0] += 1
        self.freeze()

    def rotate(self):
        """
        change the rotation of the falling piece to the next rotation, according to the numpy_figure map.
        """
        self.rotation = (self.rotation + 1) % len(self.numpy_figures[self.figure_type])
        self.figure_index = np.copy(self.numpy_figures[self.figure_type][self.rotation])

    def freeze(self):
        """Updates the field after a figure's indices have been updated"""
        self.field[tuple(self.figure_index)] = 1
        self.break_lines()
        self.new_figure()
        if np.any(self.field[(self.figure_index[0], self.figure_index[1])] == 1):
            self.state = END_OF_GAME

    def holes_score(self):
        """
        A heuristic function that scores a game state according to the number of holes it has and its bumpiness factor.
        """
        if self.state == END_OF_GAME:
            return -((len(self.field) ** 2) * len(self.field[0]))
        horizontal_variables_vector = np.zeros((len(self.field[0])), dtype=np.int8)
        horizontal_sum_vector = np.zeros((len(self.field[0])), dtype=np.int8)
        for i in range(1, len(self.field)):
            horizontal_variables_vector = np.bitwise_or(horizontal_variables_vector, self.field[i - 1])
            horizontal_sum_vector += np.bitwise_and((1 - self.field[i]), horizontal_variables_vector) * (
                    len(self.field) - i)

        vertical_sum_vector = np.zeros((len(self.field)), dtype=np.int8)
        for j in range(0, len(self.field[0]) - 1):
            vertical_sum_vector += np.bitwise_xor(self.field[:, j], self.field[:, j + 1])

        score = -np.sum(horizontal_sum_vector) - np.sum(vertical_sum_vector)
        return score

    def __hash__(self):
        return hash(str(self.field))

    def generate_all_actions(self):
        """
        :return: a list of all lists of all possible sequences of events
        """

        all_actions = []
        for value in self.generate_agent_successors_dict().values():
            all_actions += [value + [Event(pygame.KEYDOWN, pygame.K_SPACE)]]
        return all_actions

    def get_score(self):
        """Returns the score of the given state"""
        return self.score

8