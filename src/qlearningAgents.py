# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from copy import deepcopy
import pygame
import util
import features_extractor
import tetris
import numpy as np
from game_state import GameState

MAX_MOVES = 3000
END_OF_GAME = "gameover"


class QLearningAgent:

    def __init__(self, epsilon=0.05, gamma=0.75, alpha=0.2, num_training=10, num_testing=5, gui=False):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.num_training = int(num_training)
        self.num_testing = int(num_testing)
        self.gui = bool(gui)
        self.qValues = dict()

    def get_q_value(self, game: GameState, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we never seen
        a state or (state,action) tuple
        """
        relaxed_state = game.field
        figure_type = game.figure_type
        if ((relaxed_state, figure_type), action) not in self.qValues:
            self.qValues[((relaxed_state, figure_type), action)] = 0.0
            return 0.0
        else:
            return self.qValues[((relaxed_state, figure_type), action)]

    def get_value(self, state: GameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        max_q_val = float('-inf')

        for action in self.get_legal_actions(state):
            cur_q_val = self.get_q_value(state, action)
            if cur_q_val > max_q_val:
                max_q_val = cur_q_val

        if len(self.get_legal_actions(state)) == 0:
            return 0.0
        return max_q_val

    def get_legal_actions(self, game):
        """
        :param game: a tetris module
        :return: a list of all the events that the game can perform
        """
        game_state = game
        if not isinstance(game, GameState):
            game_state = GameState(game.field, game.figure.type)
        all_actions = game_state.generate_all_actions()
        return all_actions

    def get_policy(self, state: GameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        best_actions = []
        max_q_val = self.get_value(state)
        if len(self.get_legal_actions(state)) == 0:
            return None
        for action in self.get_legal_actions(state):

            if self.get_q_value(state, action) == max_q_val:
                best_actions.append(action)

        return random.choice(best_actions)

    def get_action_sequence(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        if isinstance(state, tetris.Tetris):
            state = GameState(state.field, state.figure.type)
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        if len(legal_actions) == 0:
            return None
        choice = util.flipCoin(self.epsilon)
        best_action = self.get_policy(state)
        random_action = random.choice(legal_actions)
        if choice == 1:
            action = random_action
        else:
            action = best_action
        return action

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
        """
        self.qValues[(state, action)] = self.get_q_value(state, action) + \
                                        self.alpha * (reward + self.discount * self.get_value(next_state)
                                                      - self.get_q_value(state, action))


class TetrisQAgent(QLearningAgent):
    """Exactly the same as QLearningAgent, but with different default parameters"""

    def __init__(self, epsilon=0.05, gamma=0.75, alpha=0.2, num_training=10, num_testing=5, gui=False, **args):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        args['num_testing'] = num_testing
        args['gui'] = gui
        self.index = 0
        QLearningAgent.__init__(self, **args)


class ApproximateQAgent(TetrisQAgent):

    def __init__(self, **args):
        TetrisQAgent.__init__(self, **args)
        self.counter = 0
        self.moves_tried = 0
        self.weights = dict()
        self.done = False
        self.reward_weights = [0.8, 0.2, 0.2]
        temp_game = tetris.Tetris(20, 10)
        temp_game.new_figure()
        self.game = GameState(temp_game.field, temp_game.figure.type)

        if self.num_training == 0:
            self.weights = {  # Weights found during testing
        'bias': -190.83667758428328,
        'skyline_diff': -1514.7129500869028,
        'max_skyline_diff': -2211.239718486838,
        'num_holes': -8435.39859867022,
        'max_height': -606.511815161419,
        'num_rows_cleared': 147.0848355640954}
        else:
            self.run_training_rounds()

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        if isinstance(state, tetris.Tetris):
            state = GameState(state.field, state.figure.type)
        features = features_extractor.FeatureExtractors().get_all_features(state, action)
        q_val = 0

        for feature_name in features:
            weight_val = 0 if feature_name not in self.weights.keys() else self.weights[feature_name]
            q_val += features[feature_name] * weight_val

        return q_val

    def update(self, state, action, next_state, reward):
        """Updates weights based on transition"""
        if isinstance(state, tetris.Tetris):
            state = GameState(state.field, state.figure.type)
        if isinstance(next_state, tetris.Tetris):
            next_state = GameState(next_state.field, next_state.figure.type)

        correction = np.longdouble((reward + self.discount * self.get_value(next_state)) - self.get_q_value(state, action))
        features = features_extractor.FeatureExtractors().get_all_features(state, action)
        for feature_name in features_extractor.FeatureExtractors().get_all_features(state, action):
            weight_val = 0 if feature_name not in self.weights.keys() else self.weights[feature_name]
            self.weights[feature_name] = weight_val + self.alpha * correction * features[feature_name]

    def finish_training(self):
        """call when training is complete to stop exploration """
        self.alpha = 0
        self.epsilon = 0

    def run_one_game(self, training=False, gui=False):
        """Runs one tetris game without a GUI"""
        move = 1
        while not self.done:
            if training:
                prev_game = deepcopy(self.game)
            chosen_action = self.get_action_sequence(self.game)
            for event in chosen_action:
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.game.rotate()
                    if event.key == pygame.K_LEFT:
                        self.game.push_piece_x(-1)
                    if event.key == pygame.K_RIGHT:
                        self.game.push_piece_x(1)
                    if event.key == pygame.K_SPACE:
                        self.game.push_piece_y()
            if training:
                curr_game = deepcopy(self.game)
            move += 1
            if training:
                self.observe_transition(prev_game, curr_game, chosen_action)
            if self.game.state == END_OF_GAME:
                if not gui:
                    with open('test_results.txt', 'a') as f:
                        print("game over, total moves = ", move, file=f)
                    print("game over, total moves = ", move)
                self.done = True
            if move == MAX_MOVES:
                if not gui:
                    with open('test_results.txt', 'a') as f:
                        print("Game ended because moves reached ", MAX_MOVES, file=f)
                    print("Game ended because moves reached ", MAX_MOVES)
                self.done = True
                self.game.state = END_OF_GAME

    def run_training_rounds(self):
        """Runs the number of training rounds as per the stored value"""
        scores = []
        for game_num in range(self.num_training):

            with open('test_results.txt', 'a') as f:
                print("Training Game Number %d of %d" % (game_num + 1, self.num_training), end=' ', file=f)
            print("Training Game Number %d of %d" % (game_num + 1, self.num_training), end=' ')

            self.run_one_game(training=True, gui=True)
            scores.append(self.game.get_score())

            with open('test_results.txt', 'a') as f:
                print("Score = {}".format(self.game.get_score()), file=f)
                print("approx_q_agent.weights = ", self.weights, file=f)
            print("Score = {}".format(self.game.get_score()))

            self.reset()
        return scores

    def run_testing_rounds(self, gui=False):
        """Runs the number of testing rounds as per the stored value"""
        self.finish_training()
        scores = []
        for game_num in range(self.num_testing):
            if not gui:
                with open('test_results.txt', 'a') as f:
                    print("Test Game Number %d of %d" % (game_num + 1, self.num_testing), end=' ', file=f)
                print("Test Game Number %d of %d" % (game_num + 1, self.num_testing), end=' ')
            self.run_one_game(gui=gui)
            scores.append(self.game.get_score())
            if not gui:
                with open('test_results.txt', 'a') as f:
                    print("Score = {}".format(self.game.get_score()), file=f)
                    print("approx_q_agent.weights = ", self.weights, file=f)
                print("Score = {}".format(self.game.get_score()))
                print("approx_q_agent.weights = ", self.weights)
            self.reset()
        return scores

    def reset(self):
        """Resets the game to allow for another simulation"""
        self.done = False
        new_field = []
        for _ in range(20):
            new_line = []
            for j in range(10):
                new_line.append(0)
            new_field.append(new_line)
        self.game = GameState(new_field, self.game.figure_type)

    def observe_transition(self, prev_game, curr_game, action):
        """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments
        """
        delta_reward = self.delta_reward(prev_game, curr_game)
        self.update(state=prev_game, action=action, next_state=curr_game, reward=delta_reward)

    def delta_reward(self, prev_game, curr_game):
        """
        R(S)=ω(Lp −Lc)+β(Hp −Hc)−λD2
        When L is the sum of the absolute value of the height differences between each two adjacent columns,
        H is the number of holes in the configuration and D is the absolute height of the configuration.
        D is quadratic because we wanted the absolute height to have more weight as it gets larger.
        p and c stand for previous state and current state respectively.
        """
        return self.weighted_reward(prev_game=prev_game, curr_game=curr_game, weights=self.reward_weights)

    def adversarial_search_reward(self, curr_game: GameState, prev_game: GameState):
        """
        Reward based on the reward created for the adversarial agent, as of now this reward gives us a very tall tower
        """
        import adversarial_search_agents
        return adversarial_search_agents.score_evaluation_function(curr_game) - \
               adversarial_search_agents.score_evaluation_function(prev_game)

    def weighted_reward(self, prev_game, curr_game, weights):
        """
        This reward is based on a project that tried to solve tetris with reinforment learning
        https://www.cs.huji.ac.il/~ai/projects/old/Tetris3.pdf
        :param weights: a 1x3 vector representing the weights for the features
        R(S)=ω(Lp −Lc)+β(Hp −Hc)−λD2
        When L is the sum of the absolute value of the height differences between each two adjacent columns,
        H is the number of holes in the configuration and D is the absolute height of the configuration.
        D is quadratic because we wanted the absolute height to have more weight as it gets larger.
        p and c stand for previous state and current state respectively.
        """
        features_ext = features_extractor.FeatureExtractors()
        height_dif_cur = features_ext.get_height_differences(rows=len(curr_game.field), cols=len(curr_game.field[0]), state=curr_game)
        height_dif_prev = features_ext.get_height_differences(rows=len(prev_game.field), cols=len(prev_game.field[0]), state=prev_game)

        num_holes_cur = features_ext.get_num_holes(rows=len(curr_game.field), cols=len(curr_game.field[0]), state=curr_game)
        num_holes_prev = features_ext.get_num_holes(rows=len(prev_game.field), cols=len(prev_game.field[0]), state=prev_game)

        absolute_height_cur = features_ext.get_absolute_height(rows=len(curr_game.field), cols=len(curr_game.field[0]), state=curr_game)

        reward = weights[0] * (height_dif_prev - height_dif_cur) + weights[1] * (num_holes_prev - num_holes_cur) - weights[2] * (absolute_height_cur ** 2)

        return reward