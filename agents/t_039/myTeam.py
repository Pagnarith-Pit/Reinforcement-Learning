from template import Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import random



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
        

    
    def SelectAction(self,actions,game_state):
        #self.gameRule.generateSuccessor(game_state, actions, self.id)
        # Use -p option to print to console
        self.gameRule.agent_colors = game_state.agent_colors
        
        print("My ID is: ", self.id)
        print("My start state is: ", boardToString(game_state.board, 8))
        print("Action list is: ", actions)
        

        for action in actions:
            childState = self.gameRule.generateSuccessor(game_state, action, self.id)
            print("This is the action: ", action)
            print("This is the child state: ", boardToString(childState.board, 8))

        return random.choice(actions)


