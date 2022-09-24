from template import Agent
from Reversi.reversi_model import *
import random

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
    
    def selection(self, game_state):
        
        pass

    def expansion():
        pass

    def simulation():
        pass

    def backPropogation():
        pass
        
    def SelectAction(self,actions,game_state):
        #self.gameRule.generateSuccessor(game_state, actions, self.id)
        # Use -p option to print to console
        return random.choice(actions)