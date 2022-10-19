from template import Agent
from Reversi.reversi_model import *
import math
import random

MAX = math.inf
MIN = -math.inf
GRID_SIZE = 8
THRESHOLD = 14

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
        self.validPos = self.validPos()
        self.gameRule = ReversiGameRule(2)
        self.stepCount = 0

    def SelectAction(self,actions,game_state):
        self.stepCount += 1
        """ Initialises and returns an action from the Greedy Search algorithm"""
        actions = list(set(actions))
        bestval = 0
        nextAction = random.choice(actions)
        player_id = self.id

        # If player is forced to pass
        if actions == ["Pass"]:
            return "Pass"

        action_child_states = [(action, self.gameRule.generateSuccessor(game_state, action, self.id)) for action in actions]
        
        # Play to maximise player score pre move threshold
        if self.stepCount < THRESHOLD:
            bestval = MIN
            for (action, child_state) in action_child_states:
                val = self.GreedyBF(child_state, player_id)
                if val > bestval:
                    bestval = val
                    nextAction = action
        
        # Play to minimize opponent score post move threshold
        if self.stepCount >= THRESHOLD:
            bestval = MAX
            for (action, child_state) in action_child_states:
                val = self.GreedyBF(child_state, self.Op_id(player_id))
                if val < bestval:
                    bestval = val
                    nextAction = action
        
        return nextAction

    def GreedyBF(self, game_state, agent_id):
        """ Returns heuristic value for a game state """

        return self.HeuristicScore(game_state, agent_id)

    def Op_id(self, agent_id):
        """ Returns the id of the opponent of a player """
        return (agent_id + 1) % 2

    def HeuristicScore(self, game_state, agent_id):
        """ Returns an overall heuristic value for a state """
        eval = 5 * self.ScoreHeuristic(game_state, agent_id) + 50 * self.CornerHeuristic(game_state, agent_id) \
                + 15 * self.StaticWeightHeuristic(game_state, agent_id) + 10 * self.MobilityHeuristic(game_state, agent_id) \
                + 10 * self.FrontierDiscs(game_state, agent_id)
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

    def FrontierDiscs(self,game_state,agent_id):
        """
        Discs next to empty squares. These discs have a higher liklihood of being flipped, 
        so it is ideal to minimize the number of frontier discs we control.
        """
        ownFrontiers = set()
        opFrontiers = set()

        # Compute frontier discs 
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                # Own frontier discs
                if game_state.board[x][y] == self.gameRule.agent_colors[agent_id]:
                    # Enumerate over possible directions 
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        # If a valid and empty move, disc is a frontier disc
                        if temp_pos in self.validPos and game_state.getCell(temp_pos) == Cell.EMPTY:
                            ownFrontiers.add(pos) 

                # Opponent frontier discs
                elif game_state.board[x][y] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                    # Enumerate over possible directions 
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        # If a valid and empty move, disc is a frontier disc 
                        if temp_pos in self.validPos and game_state.getCell(temp_pos) == Cell.EMPTY:
                            opFrontiers.add(pos)
        # Compute metrics 
        ownFrontierLen = len(ownFrontiers)
        opFrontierLen = len(opFrontiers)
        lenDiff = ownFrontierLen - opFrontierLen
        totalFrontier = ownFrontierLen + opFrontierLen

        # Player has more frontier discs than opponent 
        if lenDiff > 0:
            return - 100 * ownFrontierLen / totalFrontier

        # Opponent has more frontier discs than the player
        elif lenDiff < 0:
            return 100 * opFrontierLen / totalFrontier
            
        elif lenDiff == 0:
            return 0 
    def validPos(self):
        pos_list=[]
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                pos_list.append((x,y))
        return pos_list