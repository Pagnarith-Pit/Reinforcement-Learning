from template import Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import math
import random
import time

MAX_DEPTH = 3
START_DEPTH = 0
MAX = math.inf
MIN = -math.inf

# Adapted and used pseudocode by Akshay L Aradhya and Rituraj Jain from GeeksforGeeks at https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(GameRule)

    def SelectAction(self,actions,game_state):
        self.gameRule.agent_colors = game_state.agent_colours
        self.minimax(START_DEPTH, game_state, True, MIN, MAX)

    def minimax(self, depth, state, maximizingPlayer, alpha, beta):
    
        # if leaf node or game ended
        if depth == MAX_DEPTH or self.gameRule.gameEnds:
            return self.GetScore(state)
    
        if maximizingPlayer:
            best = MIN
            
            # For each possible move and child node
            legal_actions = self.gameRule.getLegalActions(self, state, self.id)
            child_states = [self.gameRule.generateSuccessor(self, state, legal_actions, self.id) for legal_actions in legal_actions] 

            for child_states in child_states:
                val = self.minimax(self, depth + 1, child_states, False, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break

                return best

        else:
            best = MAX

            # For each possible move and child node
            legal_actions = self.gameRule.getLegalActions(self, state, self.id)
            child_states = [self.gameRule.generateSuccessor(self, state, legal_actions, self.id) for legal_actions in legal_actions] 

            for child_states in child_states:
                val = self.minimax(self, depth + 1, child_states, True, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)

                # Alpha Beta Pruning
                if beta <= alpha:
                    break

                return best
    
    def GetScore(self, game_state):
        return self.gameRule.calScore(game_state, self.id) - self.gameRule.calScore(game_state, self.gameRule.getNextAgentIndex)
