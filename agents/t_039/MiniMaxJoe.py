from template import GameState, GameRule, Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import math
import random
import time

INITIAL_DEPTH = 2
CUTOFF = 0
GRID_SIZE = 8 

STABILITY_WEIGHTS = [[4,  -3,  2,  2,  2,  2, -3,  4],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [2,  -1,  0,  1,  1,  0, -1,  2],
                   [2,  -1,  0,  1,  1,  0, -1,  2],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [4,  -3,  2,  2,  2,  2, -3,  4]]

# Initial values of alpha, beta 
MAX, MIN = math.inf, -math.inf


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        # added
        self.gameRule = ReversiGameRule(2)
        self.bestAction = None
        # initialize dict for {eval: move}
    
    def SelectAction(self,actions,game_state):
        """
        Returns state from minimax strategy 
        """
        TIME_LIMIT = time.time() + 0.95
        actions = list(set(actions))

        # Initialize minimax-alpha with alpha-beta pruning with values 
        #self.gameRule.agent_colors = game_state.agent_colors
        if actions == ["Pass"]:
            return "Pass"

        depth = INITIAL_DEPTH
        # initialize alpha, beta 
        alpha = MIN 
        beta = MAX

        bestAction = random.choice(actions)
            
        action_successor_states = [(action,self.gameRule.generateSuccessor(game_state, action, self.id)) for action in actions]

        for (action, successor_state) in action_successor_states:
            score = self.MinValue(successor_state,alpha,beta,self.id,depth-1)
            if score > alpha: 
                alpha = score 
                bestAction = action

        return bestAction
    

        
        #return random.choice(actions)

    def NaiveEval(self, game_state,agent_id):
        """
        Computes a naive evaluation score of the board based on 
        number of disc current player minus number of disc of the
        opposition"""
        return self.gameRule.calScore(game_state, agent_id) \
            - self.gameRule.calScore(game_state, (agent_id + 1)%2)

    def LocationScore_quick(self,game_state, agent_id):
        score = 0 
        corners_loc = [(0,0), (7,7), (0,7), (7,0)]
        corner_trap_loc = [(1,1), (1,6), (6,1), (6,6)]
        corner_adj = [(0,1), (6,0), (1,0), (0,7), (6,0), (6,7), (7,1), (7,6)]


    
        for (i,j) in corners_loc:
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score += 4
            #score += 4
        #if game_state.board[7][7] == self.gameRule.agent_colors(agent_id):
            #score += 4
        #if game_state.board[0][7] == self.gameRule.agent_colors(agent_id):
           # score += 4

        #if game_state.board[7][0] == self.gameRule.agent_colors(agent_id):
            #score += 4
        for (i,j) in corner_trap_loc: 
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score -= 4
        for (i,j) in corner_adj:
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score -= 3
        return score 



    def LocationScore(self,game_state, agent_id):
        score = 0 
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                    score += STABILITY_WEIGHTS[i][j]
                elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                    score -= STABILITY_WEIGHTS[i][j]
        return score


    def Heuristic(self, game_state, agent_id):
        score = self.NaiveEval(game_state,agent_id) + self.LocationScore(game_state, agent_id)
        return score 


    # def countScore(board,grid_size,player_color):
    # score = 0
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if board[i][j] == player_color:
    #             score += 1
    # return score   



    # # Implement MiniMax with Alpha-Beta pruning 
    # def MaxValue(self,game_state,alpha,beta,agent_id,depth):
    #     """Returns the minimax value of the state"""
    #     # alpha: the best score for MAX along the path to state 
    #     # beta: the best score for MIN along the path to state 

    #     # IF CUTOFF-Test (have not checked end game conditions)
    #     if depth == 0: 
    #         return self.NaiveEval(game_state,agent_id)
    #         return 1 

    #     # find successors: For each s in successors(state) do

    #     ## Get legal actions (of opponent?)
    #     actions = self.gameRule.getLegalActions(self,game_state,agent_id)

    #     # check length of actions?
    #     if len(actions) == 0: 
    #         return self.MinValue(game_state,alpha,beta,agent_id,depth-1)

    #     ## apply -> get successor states 
    #     successor_states = [self.gameRule.generateSuccessor(self, game_state, action, agent_id) for action in actions]


    #     #for action, successor_state in zip(actions,successor_states):
    #         #if self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) >= alpha:
    #             #self.bestAction = action
    #             #alpha = self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) 
    #     for successor_state in successor_states:
    #         alpha = max(alpha, self.MinValue(self,successor_state,alpha,beta,agent_id,depth-1))
    #         if alpha >= beta:
    #             return beta 
    #     return alpha 
    def MaxValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == 0: 
            return self.Heuristic(game_state,agent_id)


        # find successors: For each s in successors(state) do

        ## Get legal actions (of opponent?)
        actions = self.gameRule.getLegalActions(game_state,agent_id)

        # check length of actions?
        if len(actions) == 0: 
            return self.MinValue(game_state,alpha,beta,agent_id,depth-1)

        ## apply -> get successor states 
        successor_states = [self.gameRule.generateSuccessor(game_state, action, agent_id) for action in actions]


        #for action, successor_state in zip(actions,successor_states):
            #if self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) >= alpha:
                #self.bestAction = action
                #alpha = self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) 
        for successor_state in successor_states:
            score = self.MinValue(successor_state,alpha,beta,agent_id,depth-1)
            if score > beta: 
                return score 

            if score > alpha: 
                alpha = score 
        return alpha 

    def MinValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == 0: 
            return self.Heuristic(game_state,(agent_id + 1)%2)

        # find successors: For each s in successors(state) do

        ## Get legal actions 
        op_id = self.gameRule.getNextAgentIndex()
        actions = self.gameRule.getLegalActions(game_state, op_id)

        # check length of actions?
        if len(actions) == 0: 
            return self.MaxValue(game_state,alpha,beta,agent_id,depth-1)

        successor_states = [self.gameRule.generateSuccessor(game_state, action, op_id) for action in actions]

        # loop 
        # get opponent id 
        for successor_state in successor_states:
            score = self.MaxValue(successor_state,alpha,beta,agent_id,depth-1)
            if score < alpha: 
                return score 
            if score < beta:
                beta = score 
        return beta



        #     beta = min(beta, self.MaxValue(self,successor_state,alpha,beta,agent_id,depth-1))
        #     if beta >= alpha: 
        #         return alpha 
        # return beta


