from template import Agent
from Reversi.reversi_model import *
import time
import math
import random

MAX_DEPTH = 2
START_DEPTH = 0
MIN = -math.inf
GRID_SIZE = 8
TIME_LIMIT = 0.95

STATIC_WEIGHTS = [[5,  -4,  2,  2,  2,  2, -4,  5],
                [-4, -4, -1, -1, -1, -1, -4, -4],
                [2,  -1,  1,  0,  0,  1, -1,  2],
                [2,  -1,  0,  1,  1,  0, -1,  2],
                [2,  -1,  0,  1,  1,  0, -1,  2],
                [2,  -1,  1,  0,  0,  1, -1,  2],
                [-4, -4, -1, -1, -1, -1, -4, -4],
                [5,  -4,  2,  2,  2,  2, -4,  5]]

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)

    def SelectAction(self,actions,game_state):
        """ Initialises and returns an action from the NegaMax algorithm"""
        actions = list(set(actions))
        bestval = MIN
        nextAction = random.choice(actions)
        player_id = self.id
        StartTime = time.time()

        # If player is forced to pass
        if actions == ["Pass"]:
            return "Pass"

        action_child_states = [(action, self.gameRule.generateSuccessor(game_state, action, self.id)) for action in actions]
        
        for (action, child_state) in action_child_states:
            val = self.GreedyBF(child_state, player_id, StartTime)
            if val > bestval:
                bestval = val
                nextAction = action
            
        
        return nextAction

    def GreedyBF(self, game_state, agent_id, StartTime):
        """ Returns heuristic value for a game state """

        return self.HeuristicScore(game_state, agent_id)

    def Op_id(self, agent_id):
        """ Returns the id of the opponent of a player """
        return (agent_id + 1) % 2

    def HeuristicScore(self, game_state, agent_id):
        """ Returns an overall heuristic value for a state """
        eval = 5 * self.ScoreHeuristic(game_state, agent_id) + 15 * self.CornerHeuristic(game_state, agent_id) \
                + 5 * self.StaticWeightHeuristic(game_state, agent_id) + 5 * self.MobilityHeuristic(game_state, agent_id)
        return eval

    def ScoreHeuristic(self, game_state, agent_id):
        """ Heuristic is the difference between the scores of the player and opponent"""
        return (self.gameRule.calScore(game_state, agent_id) - self.gameRule.calScore(game_state, self.Op_id(agent_id)))

    def CornerHeuristic(self, game_state, agent_id):
        """ 
        Heuristic heavily weights captured corners
        
        Corner positions are extremely powerful to capture as it cannot be flipped
        and has the potential to flip many other cells.
        """
        corner_pos = [(0,0), (0,7), (7,0), (7,7)]

        player_counter = 0
        opponent_counter = 0 

        for (x,y) in corner_pos: 
            if game_state.board[x][y] == self.gameRule.agent_colors[agent_id]:
                player_counter += 1
            elif game_state.board[x][y] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                opponent_counter += 1 

        if float(player_counter) + opponent_counter != 0:
            return 100 * (float(player_counter) - opponent_counter ) / (float(player_counter) + opponent_counter)
        else:
            return 0

    def StaticWeightHeuristic(self, game_state, agent_id):
        """
        Predetermined, static weights for each cell which denotes relative strength
        of having captured any particular cell.
        Values adapted from those published by Sannidhanam and Annamalai.
        """
        player_value = 0
        opponent_value = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if game_state.board[x][y] == self.gameRule.agent_colors[agent_id]:
                    player_value += STATIC_WEIGHTS[x][y]
                elif game_state.board[x][y] == self.gameRule.agent_colors[self.Op_id(agent_id)]:
                    opponent_value += STATIC_WEIGHTS[x][y]
                
        return player_value - opponent_value

    def MobilityHeuristic(self, game_state, agent_id):
        """ Heuristic aims to restrict possible moves available to the opponent """
        player_moves = 0
        opponent_moves = 0

        # Substracting one possible move each to exclude passing
        player_moves = len(self.gameRule.getLegalActions(game_state, agent_id)) - 1
        opponent_moves = len(self.gameRule.getLegalActions(game_state, self.Op_id(agent_id))) - 1

        if player_moves != 0:
            return 100.0 * (float(player_moves) - opponent_moves) / (float(player_moves) + opponent_moves)
        
        else:
            return 0

    def TerminalState(self, game_state, player_actions, agent_id):
        """
        Checks whether the game state is a terminal state.
        Uses the list of player actions generated earlier in the NegaMax algorithm.
        """
        opponent_actions = self.gameRule.getLegalActions(game_state, self.Op_id(agent_id))
        return (player_actions == ["Pass"]) and (opponent_actions == ["Pass"])