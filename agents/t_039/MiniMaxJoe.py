from template import GameState, GameRule, Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import math
import random
import time

INITIAL_DEPTH = 3 
CUTOFF = 0

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

        # Initialize minimax-alpha with alpha-beta pruning with values 
        self.gameRule.agent_colors = game_state.agent_colors

        return self.AlphaBetaMiniMax(game_state,game_state,actions)
        #return random.choice(actions)

    def NaiveEval(self, game_state):
        """
        Computes a naive evaluation score of the board based on 
        number of disc current player minus number of disc of the
        opposition"""
        return self.gameRule.calScore(game_state, self.id) \
            - self.gameRule.calScore(game_state, (self.id + 1)%2)



    def AlphaBetaMiniMax(self, game_state,agent_id,actions):
        depth = INITIAL_DEPTH
        # initialize alpha, beta 
        alpha, beta = MIN, MAX 

        action_successor_states = [(action,self.gameRule.generateSuccessor(self, game_state, action, agent_id)) for action in actions]

        for (action, successor_state) in action_successor_states:
            score = self.MinValue(successor_state,alpha,beta,agent_id,depth-1)
            if score > alpha: 
                alpha = score 
                self.bestAction = action

        return self.bestAction


    # Implement MiniMax with Alpha-Beta pruning 
    def MaxValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == CUTOFF: 
            return self.NaiveEval(game_state)

        # find successors: For each s in successors(state) do

        ## Get legal actions 
        actions = self.gameRule.getLegalActions(self, game_state, agent_id)

        # check length of actions?
        if len(actions) == 0: 
            return self.MinValue(game_state,alpha,beta,self.gameRule.getNextAgentIndex(),depth-1)

        ## apply -> get successor states 
        successor_states = [self.gameRule.generateSuccessor(self, game_state, action, agent_id) for action in actions]

        # loop 
        # get opponent id 
        op_id = self.gameRule.getNextAgentIndex()
        #for action, successor_state in zip(actions,successor_states):
            #if self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) >= alpha:
                #self.bestAction = action
                #alpha = self.MinValue(self,successor_state,alpha,beta,op_id,depth-1) 
        for successor_state in successor_states:
            alpha = max(alpha, self.MinValue(self,successor_state,alpha,beta,op_id,depth-1))
            if alpha >= beta:
                return beta 
        return alpha 

    def MinValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == CUTOFF: 
            return self.NaiveEval(game_state)

        # find successors: For each s in successors(state) do

        ## Get legal actions 
        actions = self.gameRule.getLegalActions(self, game_state, agent_id)

        # check length of actions?
        if len(actions) == 0: 
            return self.MaxValue(game_state,alpha,beta,self.gameRule.getNextAgentIndex(),depth-1)

        ## apply -> get successor states 
        successor_states = [self.gameRule.generateSuccessor(self, game_state, action, agent_id) for action in actions]

        # loop 
        # get opponent id 
        op_id = self.gameRule.getNextAgentIndex()
        for successor_state in successor_states:
            beta = min(beta, self.MaxValue(self,successor_state,alpha,beta,op_id,depth-1))
            if beta >= alpha: 
                return alpha 
        return beta


