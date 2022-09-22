from template import Agent
from Reversi.reversi_model import *
import random



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        #self.gameRule = ReversiGameRule()
        
    
    def SelectAction(self,actions,game_state):
        #self.gameRule.generateSuccessor(game_state, actions, self.id)
        print(game_state.num)
        return random.choice(actions)


