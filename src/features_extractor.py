import numpy as np

from util import Counter
from copy import deepcopy
import pygame
from game_state import GameState


class FeatureExtractors:
    """Extracts features from a (state, action) pair"""

    def get_all_features(self, state, action):
        """
          Returns a Counter from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        if state.__class__.__name__ == 'Tetris':
            state = GameState(state.field, state.figure.type)
        successor_state = deepcopy(state)
        # simulate action on state
        for event in action:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    successor_state.rotate()
                if event.key == pygame.K_LEFT:
                    successor_state.push_piece_x(-1)
                if event.key == pygame.K_RIGHT:
                    successor_state.push_piece_x(1)
                if event.key == pygame.K_SPACE:
                    successor_state.push_piece_y()

        rows, cols = len(state.field), len(state.field[0])
        features = Counter()
        features["bias"] = 1.0
        features["skyline_diff"] = self.get_height_differences(rows, cols, successor_state)
        features["max_skyline_diff"] = self.max_height_diff(rows, cols, successor_state)
        features["num_holes"] = self.get_num_holes(rows, cols, successor_state)
        features["max_height"] = self.get_max_height(rows, cols, successor_state)
        features["num_rows_cleared"] = successor_state.score - state.score

        features.divideAll(10.0)
        return features

    def max_height_diff(self, rows, cols, state):
        """
        :return: The difference between the max and min skyline height
        """
        skyline = self.get_skyline(rows, cols, state)
        return np.max(skyline) - np.min(skyline)

    def num_rows_cleared(self, state, rows):
        """
        :return: True iff applying this action would yield in clearing a row
        """
        count = 0
        for row in range(rows):
            if np.all(state.field[row] == 1):
                count += 1
        return count

    def get_max_height(self, rows, cols, state):
        """
        :return: The row of the highest filled square
        """
        max_col = np.min(self.get_skyline(rows, cols, state))
        return 20 - max_col

    def get_skyline(self, rows, cols, state):
        """
        :return: Returns a Numpy array with the index of the highest filled square in each column
        """
        output = np.full(cols, fill_value=rows)
        for col in range(cols):
            for row in range(rows):
                if state.field[row][col] == 1:
                    output[col] = row
                    break
        return output

    def get_num_holes(self, rows, cols, state):
        """
        :return: The number of holes in the board. A hole is defined by an empty square that is below the skyline
                 A hole is defined as an empty square that rests below the skyline
        """
        count = 0
        for col in range(cols):
            below_skyline = False
            for row in range(rows):
                if state.field[row][col] == 0:
                    if below_skyline:
                        count += 1
                elif not below_skyline:  # We have found the first filled square in the column
                    below_skyline = True
        return count

    def get_height_differences(self, rows, cols, state):
        """
        :return: The sum of the differences between adjacent skyline tiles
        """
        total = 0
        skyline = self.get_skyline(rows, cols, state)
        for col in range(cols - 1):
            total += abs(skyline[col] - skyline[col + 1])
        return total

    def get_absolute_height(self, rows, cols, state):
        """
        :param rows: The number of rows in the board
        :param cols: The number of columns in the board
        :param state: A GameState object
        :return: The sum of the heights of each column
        """

        skyline = self.get_skyline(rows, cols, state)
        sum = 0

        for i in range(len(skyline)):
            sum += (20 - skyline[i])
        return sum
