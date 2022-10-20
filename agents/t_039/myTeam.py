from template import Agent
from Reversi.reversi_model import *
import math
import random
import time
from agents.t_039.MCTS import Node, MCTS

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
        self.MCTS = MCTS(self.gameRule, _id)

    def SelectAction(self,actions,game_state):

        if actions == ["Pass"]:
            return "Pass"
        print("Newest Agent")
        self.gameRule.agent_colors = game_state.agent_colors

        # if self.stepCount > 20:
        #     action = self.MM.SelectAction(actions,game_state)
        #     return action

        root = self.MCTS.run(game_state)

        # if self.stepCount > 10:
        #     print("\nNew Line")
        #     for action, child in root.children.items():
        #         print("This is the action: ", action)
        #         print("This is child count: ", child.visit_count)
        #         print("This is child value: ", child.value_sum)

        action = root.select_action()
        return action
