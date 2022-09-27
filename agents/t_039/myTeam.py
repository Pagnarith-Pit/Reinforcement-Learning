from template import Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import random



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
        

    
    def SelectAction(self,actions,game_state):
        return random.choice(actions)


