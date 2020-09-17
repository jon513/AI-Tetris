import game_state
import abc
import util
import numpy as np

MAX_PLAYER = 1

MIN_PLAYER = 0

ACTION = 1

STATE = 0


class AdversarialSearchAgent:
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the AlphaBetaAgent & ExpectimaxAgent.
    """

    def __init__(self, evaluation_function='score_evaluation_function', depth=1):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        self.expanded_nodes = 0

    @abc.abstractmethod
    def get_action_sequence(self, tetris_game_state):
        return


class ExpectimaxAgent(AdversarialSearchAgent):
    """
    An agent that uses the expectimax algorithm in order to evaluate and return a course of action in a given game
    state. The score is given by the method holes_score of the game_state objects and its details are depicted there.
    """

    def get_action_sequence(self, tetris_game):
        """
        Given a Tetris object that represents the current state of the game, return a list of Event objects that
        represents a course of action.
        """

        self.expanded_nodes = 0
        best_score = -np.inf
        best_action_sequence = []
        tetris_game_state = game_state.GameState(tetris_game.field, tetris_game.figure.type)
        successors_dict = tetris_game_state.generate_agent_successors_dict()  # maps a game state to an event list.

        for successor in successors_dict.keys():
            value = self.expected_value(successor, 1)
            if value >= best_score:
                best_score = value
                best_action_sequence = successors_dict[successor]
        return best_action_sequence

    def expected_value(self, tetris_game_state, depth):
        # base case - if the depth is reached return the value computed by the evaluation function:
        self.expanded_nodes += 1
        if depth >= self.depth:
            return self.evaluation_function(tetris_game_state)

        successors_list = tetris_game_state.generate_opponent_successors_list()
        p = 1 / len(successors_list)
        value = 0
        for successor in successors_list:
            value += p * (self.max_value(successor, depth + 0.5))

        return value

    def max_value(self, tetris_game_state, depth):
        # base case - if the depth is reached return the value computed by the evaluation function:
        if depth >= self.depth:
            return self.evaluation_function(tetris_game_state)
        self.expanded_nodes += 1
        value = -np.inf
        successors_list = tetris_game_state.generate_agent_successors_list()
        for successor in successors_list:
            value = max(value, self.expected_value(successor, depth + 0.5))
        return value

    def __str__(self):
        return "Expectimax Agent"


class AlphaBetaAgent(AdversarialSearchAgent):
    """
    An agent that uses the Min-Max algorithm for choosing the best action that the player perform, and improve this
    computation time by using the Alpha-Beta pruning algorithm.
    """

    def get_sorted_successors_list(self, tetris_game_state, player):
        """
        generates successors and sorts them according to the current player
        """
        if player == MAX_PLAYER:
            successors = tetris_game_state.generate_agent_successors_list()
        else:
            successors = tetris_game_state.generate_opponent_successors_list()
        sorted(successors, key=lambda tetris_state: self.evaluation_function(tetris_state), reverse=player)
        return successors

    def get_action_sequence(self, tetris_game):
        """
        Returns the minimax action
        """
        self.expanded_nodes = 0
        best_score = -np.inf
        alpha = -np.inf
        beta = np.inf
        best_action_sequence = []
        tetris_game_state = game_state.GameState(tetris_game.field, tetris_game.figure.type)
        successors_dict = tetris_game_state.generate_agent_successors_dict()  # maps a game state to an event list.

        for successor in successors_dict.keys():
            value = self.max_value(successor, 1, alpha, beta)
            if value >= best_score:
                best_score = value
                best_action_sequence = successors_dict[successor]
        return best_action_sequence

    def max_value(self, tetris_game_state, depth, alpha, beta):
        # base case - if the depth is reached return the value computed by the evaluation function:
        if depth >= self.depth:
            return self.evaluation_function(tetris_game_state)
        self.expanded_nodes += 1
        value = -np.inf
        successors_list = tetris_game_state.generate_agent_successors_list()
        for successor in successors_list:
            value = max(value, self.min_value(successor, depth + 0.5, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, tetris_state, depth, alpha, beta):
        # base case - if the depth is reached return the value computed by the evaluation function:
        if depth >= self.depth:
            return self.evaluation_function(tetris_state)
        value = np.inf  # set value to minus infinity.
        sorted_states = self.get_sorted_successors_list(tetris_state, MIN_PLAYER)
        for state in sorted_states:
            value = min(value, self.max_value(state, depth + 0.5, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def __str__(self):
        return "Alpha-Beta agent"


def score_evaluation_function(tetris_game_state):
    """
    Use the holes_score method of the GameState class in order to evaluate the given state
    """
    return tetris_game_state.holes_score()