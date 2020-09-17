import pygame
from tetris import Tetris
import adversarial_search_agents
from qlearningAgents import ApproximateQAgent


class Event:
    type = None
    key = None

    def __init__(self, type, key):
        self.type = type
        self.key = key


counter = 0


def run_ai(game: Tetris, heuristic=None):
    global counter
    counter += 1
    if counter < 3:
        return []
    counter = 0
    e = Event(pygame.KEYDOWN, pygame.K_UP)
    return [e]


def create_agent(name, args_dict):
    if name == "expectimax":
        return adversarial_search_agents.ExpectimaxAgent(depth=args_dict['depth'])
    elif name == "alpha_beta":
        return adversarial_search_agents.AlphaBetaAgent(depth=args_dict['depth'])
    elif name == "q_learning":
        return ApproximateQAgent(
            alpha=args_dict['alpha'],
            gamma=args_dict['gamma'],
            epsilon=args_dict['epsilon'],
            num_training=args_dict['num_training'],
            num_testing=args_dict['num_testing'],
            gui=True
        )


def get_ai_agent(agent_name, args_dict):
    agent = create_agent(agent_name, args_dict)

    def inner_adversarial(_game):
        if _game.figure.x == 3 and _game.figure.y == 0:
            return agent.get_action_sequence(_game)
        else:
            return [Event(pygame.KEYDOWN, pygame.K_DOWN)]

    def inner_q_learning(_game):
        global counter
        counter += 1
        if counter < 3:
            return []
        counter = 0
        return agent.get_action_sequence(_game)

    return inner_q_learning if agent_name == 'q_learning' else inner_adversarial
