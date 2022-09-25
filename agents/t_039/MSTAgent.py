from template import Agent
from Reversi.reversi_model import *
import math
import random

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
    
    def selection(self, parent_state, game_state_child_list, visited_state_dict):
        maxUCB = 0
        selected = game_state_child_list[0]
        parent_num_of_visit = visited_state_dict[parent_state][1]

        for state in game_state_child_list:
            total_score, num_of_visit = visited_state_dict[state]
            UCB_val = (total_score/num_of_visit) + 2*math.sqrt(math.log(parent_num_of_visit)/num_of_visit)

            if UCB_val > maxUCB:
                maxUCB = UCB_val
                selected = state
        
        return selected

    def expansion(self, game_state, action, id, visited_state_dict):
        total_score, num_of_vist = visited_state_dict[game_state]
        num_of_vist += 1

        visited_state_dict[game_state] = (total_score, num_of_vist)
        pass

    def simulation():
        pass

    def backPropogation():
        pass
        
    def SelectAction(self,actions,game_state):
        #self.gameRule.generateSuccessor(game_state, actions, self.id)
        # Use -p option to print to console
        ITERATION = 1000
        visited = dict()

        for i in range(ITERATION):
            selectedState = []

            node = game_state
            rootNode = False
            while not rootNode:
                expansion()
                selection()
            pass
        return random.choice(actions)