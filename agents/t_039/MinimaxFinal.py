from datetime import timedelta
from template import GameState, GameRule, Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString, Cell
import math
import random
import time
import operator



# Initial values of alpha, beta 



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        # added
        self.GRID_SIZE = 8
        self.MAX, self.MIN = math.inf, -math.inf
        self.prevBoard = [[Cell.EMPTY for i in range(self.GRID_SIZE)] for i in range(self.GRID_SIZE)]
        self.prevBoard[4-1][4-1] = Cell.WHITE
        self.prevBoard[4][4-1] = Cell.BLACK
        self.prevBoard[4-1][4] = Cell.BLACK
        self.prevBoard[4][4] = Cell.WHITE

        self.playHistory = ''
        self.openBook = False
        self.validPos = self.validPos()
        self.gameRule = ReversiGameRule(2)
        self.bestAction = None
        self.op_id = self.gameRule.getNextAgentIndex()
        self.playHistory = ''
        self.INITIAL_DEPTH = 8
        self.TIME_LIMIT = 0.78
        self.MAX_DISC = 64
        self.STABILITY_WEIGHTS = [[4,  -3,  2,  2,  2,  2, -3,  4],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [2,  -1,  0,  0,  0,  0, -1,  2],
                   [2,  -1,  0,  0,  0,  0, -1,  2],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [4,  -3,  2,  2,  2,  2, -3,  4]]

        self.DISC_SQUARES =    [[20,  -4,  11,  8,   8,  11,  -4,   20],
                   [-4,  -7,  -4,  -1,  -1, -4,   -8,  -4],
                   [11,  -4,   2,   2,   2,  2,   -4,  11],
                   [8,    -1,  2, -3, -3,  2,    -1,   8],
                   [8,   -1,   2,  -3,  -3,  2,    -1,  8],
                   [11,  -4,   2,   2,   2,  2,   -4,   11],
                   [-4,  -8,  -4,  -1,  -1, -4,   -8,  -4],
                   [20,  -4,   11,  8,   8, 11,   -4,   20]]
        self.COL_MAPPING= {0:'a', 1:'b', 2: 'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
 
        # initialize dict for {eval: move}

        # Debugging / testing 
        self.statesExpanded = 0 



    def SelectAction(self,actions,game_state):
        """
        AlphaBeta MiniMax with a fixed depth of 2 
        """
        # Initialize 
        self.startTime = time.time()
        self.statesExpanded = 0 


        # Try to read from board 
        self.playHistory += self.readMove(self.prevBoard,game_state.board)
        #print(f"History before: {self.playHistory}")
        


        bestAction, bestScore = random.choice(actions), self.MIN

        self.gameRule.agent_colors = game_state.agent_colors
        if actions == ["Pass"]:
            return "Pass"

        initial_depth = self.INITIAL_DEPTH
        

        # Implement end-game check (MAX_DISCS - current discs)
        blanks = self.MAX_DISC - self.gameRule.calScore(game_state,self.id) - self.gameRule.calScore(game_state,(self.id + 1)%2)

        if blanks > 44:
            self.earlyGame = True 
        else:
            self.earlyGame = False 

        if 10 < blanks <= 44:
            self.midGame = True
        else:
            self.midGame = False 



        if blanks <= 10:
            initial_depth = blanks 
            self.endGame = True
        else:
            self.endGame = False

        if blanks <= 15:
            initial_depth = 10


        # Open book 
        # (ROW, COL) E.G (3,2) -> C4
        # -> (COL_DICT[COL], ROW+1)
        if blanks <= 60 and (3,2) in actions:
            self.openBook = True 
            self.playHistory += f"{self.COL_MAPPING[2]}{3+1}"
            nextState = self.gameRule.generateSuccessor(game_state, (3,2), self.id)
            self.prevBoard = nextState.board
            return (3,2)

        if self.openBook:
            tentative = self.getOpenMoves(self.playHistory)
            if tentative:
                #print("Open book!")
                bestAction = tentative

                nextState = self.gameRule.generateSuccessor(game_state, bestAction, self.id)
                self.prevBoard = nextState.board

                (row,col) = bestAction 
                self.playHistory += f"{self.COL_MAPPING[col]}{row+1}"
                # update things

                return bestAction

        # Check killer moves: corners 
        tentative = self.killerMove(game_state,actions,self.id)
        if tentative:
            #print("Killer move!")
            return tentative
        
        # Initialize TT table 
        self.TT = dict()



        # do this while we dont run out of time 
        for depth in range(2,initial_depth,1):
            # try to extract action, score from alpha beta 
            #action, score = self.AlphaBeta(actions,game_state,depth)

            action,score = self.AlphaBeta(actions,game_state,depth)
            
            # check whether the IDAB has terminated with success
            if action:
                bestAction = action 
                bestScore = score 

        #print(f"NewDepth6: Heurustic value: {round(bestScore,2)}; States expanded in search: {self.statesExpanded}, Time taken: {round(time.time()-self.startTime,4)}")
        nextState = self.gameRule.generateSuccessor(game_state, bestAction, self.id)
        self.prevBoard = nextState.board

        # Increment history 
        (x,y) = bestAction 
        self.playHistory += f"{self.COL_MAPPING[x]}{y+1}"

        return bestAction

        
    def timeLeft(self):
        """
        Returns True if time within bound if not False 
        """
        if time.time() - self.startTime < self.TIME_LIMIT:
            return True 

        return False
    
    def AlphaBeta(self,actions,game_state,depth):
        """
        AlphaBeta MiniMax with a specified depth
        """

        # clear 
        self.statesExpanded = 0 
        # sometimes we are provided with duplicate actions 
        actions = list(set(actions))

        # for deeper levels: move heuristics 
        if depth > 2:
            actions = list(dict(sorted(self.TT.items(), key=operator.itemgetter(1),reverse=True)).keys())

        # Initialize minimax-alpha with alpha-beta pruning with values 
        self.gameRule.agent_colors = game_state.agent_colors
        if actions == ["Pass"]:
            return "Pass"

        # initialize alpha, beta 
        alpha = self.MIN 
        beta = self.MAX

        # initialize action 
        bestAction = None 

        # get action, successor states and pass to min player 
        action_successor_states = [(action,self.gameRule.generateSuccessor(game_state, action, self.id)) for action in actions]
        for (action, successor_state) in action_successor_states:
            if not self.timeLeft():
                # we wouldn't want this value if minimax not searched fully 
                return None, alpha
            self.statesExpanded += 1 
            score = self.MinValue(successor_state,alpha,beta,self.id,depth-1)
            # store move ordering, later used by Min, Max 
            self.TT[action] = score 
            
            if not self.timeLeft():
                return None, alpha 
                
            if score > alpha: 
                alpha = score 
                bestAction = action
        
        
        return bestAction, alpha 
        #return random.choice(actions)

    def MaxValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == 0  or self.endState(game_state) or not self.timeLeft():
            return self.newEval(game_state,agent_id)
            return self.Heuristic(game_state,agent_id)

        ## Get legal actions (of opponent?)
        actions = self.gameRule.getLegalActions(game_state,agent_id)
        successor_states = [self.gameRule.generateSuccessor(game_state, action, agent_id) for action in actions]


        for successor_state in successor_states:
            # increment 
            self.statesExpanded += 1 
            # continue 
            alpha = max(alpha,self.MinValue(successor_state,alpha,beta,agent_id,depth-1))
            if alpha >= beta: 
                return beta 
        return alpha 


    def MinValue(self,game_state,alpha,beta,agent_id,depth):
        """Returns the minimax value of the state"""
        # alpha: the best score for MAX along the path to state 
        # beta: the best score for MIN along the path to state 

        # IF CUTOFF-Test (have not checked end game conditions)
        if depth == 0 or self.endState(game_state) or not self.timeLeft():
            return self.newEval(game_state,agent_id)
            return self.Heuristic(game_state,agent_id)
            return self.getStability(game_state,agent_id)
        ## Get legal actions 
        op_id = self.gameRule.getNextAgentIndex()
        actions = self.gameRule.getLegalActions(game_state, op_id)

        successor_states = [self.gameRule.generateSuccessor(game_state, action, op_id) for action in actions]

        # get opponent id 
        for successor_state in successor_states:
            # increment 
            self.statesExpanded += 1 
            # continue 
            beta = min(beta,self.MaxValue(successor_state,alpha,beta,agent_id,depth-1))
            if beta <= alpha: 
                return alpha 
        return beta 

    def readMove(self,prev_board,curr_board):
        colMapping = {0:'a', 1:'b', 2: 'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
    

        # compare: 
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                # find difference
                if prev_board[i][j] == Cell.EMPTY and curr_board[i][j] != Cell.EMPTY:
                    #print(f"opponent played (i,j):  ({j},{i})")
                    #print(f"opponent played: {colMapping[j]}{i+1}")
                    return f'{colMapping[j]}{i+1}'

        # an empty board, so return empty string 
        return ''




    def CoinParity(self, game_state,agent_id):
        """
        Computes a scaled CoinParity score that captures the difference
        between the max player and the min player.
        """
        numerator = self.gameRule.calScore(game_state, agent_id) \
            - self.gameRule.calScore(game_state, (agent_id + 1)%2)

        denominator = self.gameRule.calScore(game_state, agent_id) +\
             self.gameRule.calScore(game_state, (agent_id + 1)%2)
        return 100 * float(numerator) / float(denominator)


    def getActualMobility(self,game_state,agent_id):
        """
        Heuristic based on restricting your opponent's mobility and to mobalize
        yourself. Mobility comes in two flavors: (i) actual mobility, (ii) potential 
        mobility 
        """
        ownmoves = 0 
        opponentmoves = 0 
        # get moves, -1 to exclude pass moves 
        ownmoves = len(self.gameRule.getLegalActions(game_state, agent_id))-1
        opponentmoves = len(self.gameRule.getLegalActions(game_state,(agent_id + 1)%2))-1

        if ownmoves + opponentmoves != 0:
            MobilityHeuristic = 100.0 * (float(ownmoves)-opponentmoves) / (float(ownmoves) +opponentmoves )

        else:
            MobilityHeuristic = 0 

        return MobilityHeuristic


    def getStability(self,game_state,agent_id):
        my_stability, tot = 0,0
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                    my_stability += self.STABILITY_WEIGHTS[i][j]
                    tot += abs(self.STABILITY_WEIGHTS[i][j])
                elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                    my_stability -= self.STABILITY_WEIGHTS[i][j]
                    tot += abs(self.STABILITY_WEIGHTS[i][j])

        if not tot == 0:
            return 100 * float(my_stability) / tot
        else:
            return 0

    def getCorners(self,game_state,agent_id):
        corners_loc = [(0,0), (0,7), (7,0), (7,7)]

        self_corner,opponent_corner = 0,0 

        for (i,j) in corners_loc: 
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                self_corner += 1
            elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                opponent_corner += 1 

        if float(self_corner) + opponent_corner != 0:
            return 100 * (float(self_corner) - opponent_corner) / (float(self_corner) + opponent_corner)
        else:
            return 0


    def Heuristic(self, game_state, agent_id):
        """
        Equally weighted heuristic 
        """

        if self.end:
            eval = 100 * self.getCorners(game_state,agent_id) + 30 * self.getActualMobility(game_state,agent_id) + \
            40 * self.CoinParity(game_state,agent_id) + 30 * self.getStability(game_state,agent_id)
        elif self.beginning:
            eval = 100 * self.getCorners(game_state,agent_id) + 40 * self.getActualMobility(game_state,agent_id) + \
            25 * self.CoinParity(game_state,agent_id) + 40 * self.getStability(game_state,agent_id)

        else:
            eval = 100 * self.getCorners(game_state,agent_id) + 40 * self.getActualMobility(game_state,agent_id) + \
            40 * self.CoinParity(game_state,agent_id) + 40 * self.getStability(game_state,agent_id)


        
        return eval

    def LocationScore_quick(self,game_state, agent_id):
        score = 0 
        corners_loc = [(0,0), (7,7), (0,7), (7,0)]
        corner_trap_loc = [(1,1), (1,6), (6,1), (6,6)]
        corner_adj = [(0,1), (6,0), (1,0), (0,7), (6,0), (6,7), (7,1), (7,6)]
    
        for (i,j) in corners_loc:
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score += 4
        for (i,j) in corner_trap_loc: 
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score -= 4
        for (i,j) in corner_adj:
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                score -= 3
        return score 

    def endState(self, game_state):
        """
        Returns true/false, whether both players have action of "pass" 
        """
        own_actions = self.gameRule.getLegalActions(game_state,self.id)
        op_actions = self.gameRule.getLegalActions(game_state,self.op_id)

        return own_actions == ["Pass"] and op_actions == ["Pass"]

    #Implement opening moves 

    def getOpenMoves(self,moveHistory):
        alphaToNum = {'a':0, 'b':1,'c':2,'d':3,'e':4, 'f':5,'g':6,'h':7}
        open_moves = [
            "C4c3",
            "C4c3D3c5B2",
            "C4c3D3c5B3",
            "C4c3D3c5B3f3",
            "C4c3D3c5B3f4B5b4C6d6F5",
            "C4c3D3c5B4",
            "C4c3D3c5B4d2C2f4D6c6F5e6F7",
            "C4c3D3c5B4d2D6",
            "C4c3D3c5B4d2E2",
            "C4c3D3c5B4e3",
            "C4c3D3c5B5",
            "C4c3D3c5B6",
            "C4c3D3c5B6c6B5",
            "C4c3D3c5B6e3",
            "C4c3D3c5D6",
            "C4c3D3c5D6e3",
            "C4c3D3c5D6f4B4",
            "C4c3D3c5D6f4B4b6B5c6B3",
            "C4c3D3c5D6f4B4b6B5c6F5",
            "C4c3D3c5D6f4B4c6B5b3B6e3C2a4A5a6D2",
            "C4c3D3c5D6f4B4e3B3",
            "C4c3D3c5D6f4F5",
            "C4c3D3c5D6f4F5d2",
            "C4c3D3c5D6f4F5d2B5",
            "C4c3D3c5D6f4F5d2G4d7",
            "C4c3D3c5D6f4F5e6C6d7",
            'C4c3D3c5D6f4F5e6F6',
            'C4c3D3c5F6',
            'C4c3D3c5F6e2C6',
            'C4c3D3c5F6e3C6f5F4g5',
            'C4c3D3c5F6f5',
            'C4c3E6c5',
            'C4c3F5c5',
            'C4c5',
            'C4e3',
            'C4e3F4c5D6e6',
            'C4e3F4c5D6f3C6',
            'C4e3F4c5D6f3D3',
            'C4e3F4c5D6f3D3c3',
            'C4e3F4c5D6f3E2',
            'C4e3F4c5D6f3E6c3D3e2',
            'C4e3F4c5D6f3E6c3D3e2B5',
            'C4e3F4c5D6f3E6c3D3e2B5f5',
            'C4e3F4c5D6f3E6c3D3e2B5f5B3',
            'C4e3F4c5D6f3E6c3D3e2B5f5B4f6C2e7D2c7',
            'C4e3F4c5D6f3E6c3D3e2B6f5',
            'C4e3F4c5D6f3E6c3D3e2B6f5B4f6G5d7',
            'C4e3F4c5D6f3E6c3D3e2B6f5G5',
            'C4e3F4c5D6f3E6c3D3e2B6f5G5f6',
            'C4e3F4c5D6f3E6c3D3e2D2',
            'C4e3F4c5D6f3E6c6',
            'C4e3F4c5E6',
            'C4e3F5b4',
            'C4e3F5b4F3',
            'C4e3F5b4F3f4E2e6G5f6D6c6',
            'C4e3F5e6D3',
            'C4e3F5e6F4',
            'C4e3F5e6F4c5D6c6F7f3',
            'C4e3F5e6F4c5D6c6F7g5G6',
            'C4e3F6b4',
            'C4e3F6e6F5',
            'C4e3F6e6F5c5C3',
            'C4e3F6e6F5c5C3b4',
            'C4e3F6e6F5c5C3b4D6c6B5a6B6c7',
            'C4e3F6e6F5c5C3c6',
            'C4e3F6e6F5c5C3c6D3d2E2b3C1c2B4a3A5b5A6a4A2',
            'C4e3F6e6F5c5C3c6D6',
            'C4e3F6e6F5c5C3g5',
            'C4e3F6e6F5c5D3',
            'C4e3F6e6F5c5D6',
            'C4e3F6e6F5c5F4g5G4f3C6d3D6',
            'C4e3F6e6F5c5F4g5G4f3C6d3D6b3C3b4E2b6',
            'C4e3F6e6F5c5F4g6F7',
            'C4e3F6e6F5c5F4g6F7d3',
            'C4e3F6e6F5c5F4g6F7g5',
            'C4e3F6e6F5g6',
            'C4e3F6e6F5g6E7c5'

        ]

        # now compare string 

        lenMoveHistory = len(moveHistory)
        for move in open_moves:
            lenMove = len(move)
            lower = move.lower()
            if lenMove >= (lenMoveHistory+2) and lower.startswith(moveHistory):

                alpha_move = lower[lenMoveHistory:lenMoveHistory+2]
                col = alpha_move[0]
                row = alpha_move[1]
                #print(f'alpha move is: {alpha_move}')
                #print(f"going to return this..({int(row)-1},{alphaToNum[col]} ")
                return (int(row)-1,alphaToNum[col])

        self.openBook = False 
        return None 


    def killerMove(self,game_state,actions,agent_id):
        # check if corner within available actions 
        corners_loc = [(0,0), (7,7), (0,7), (7,0)]

        potential = []
        killer_dict = dict()

        for corner in corners_loc:
            if corner in actions:
                potential.append(corner)
        # just directly evaluate from here

        if potential: 
            return potential[0]

        ## Find blocking move 
        successor_states = [self.gameRule.generateSuccessor(game_state, action, agent_id) for action in actions]

        # blocking move 
        for action, successor_state in zip(actions,successor_states):
            if len(self.gameRule.getLegalActions(successor_state, (agent_id + 1)%2))-1 == 0:
                #print('Blocking move!')
                return action 

        return None
       # Implement Evaluator (MLia)
    def pieceDifference(self,game_state,agent_id):
        """
        Measures how many pieces of each color are on the baord. 
        - Count the number of black and pieces and white pieces on the board
        - If the number of black pieces B is greater than the number of white pieces: 
            - p = 100 B / (B+W) 
        - If number of white pieces W is greater than the number of black pieces 
            - p = 100 W / (B+W)
        - If B = W 
            - p = 0                
        """
        ownScore = self.gameRule.calScore(game_state, agent_id)
        oppScore = self.gameRule.calScore(game_state, (agent_id + 1)%2)
        scoreDiff = ownScore - oppScore
        denom = ownScore + oppScore

        if scoreDiff > 0: 
            return 100 * ownScore / denom
        elif scoreDiff < 0: 
            return -100 * oppScore / denom
        else: 
            return 0 

    def cornerOccupancy(self,game_state,agent_id):
        """
        Corners are the most valuable squares on the board
        Measures how many corners are owned by each player
        """
        corners_loc = [(0,0), (0,7), (7,0), (7,7)]
        self_corner,opponent_corner = 0,0 

        for (i,j) in corners_loc: 
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                self_corner += 1
            elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                opponent_corner += 1

        return 25 * (self_corner - opponent_corner)

    
    def cornerCloseness(self,game_state,agent_id):
        """"
        Squares adjacent to the corners can be deadly if the corner is empty 
        Measures pieces adjacent to empty corners
        """

        adj_corners_loc = [(0,1), (1,0), (1,1), (0,6), (1,7), (1,6),
        (7,1),(6,0),(6,1), (7,6), (6,7), (6,6)]

        self_adj_corner,opponent_adj_corner = 0,0 

        for (i,j) in adj_corners_loc: 
            if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                self_adj_corner += 1
            elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                opponent_adj_corner += 1

        return -12.5 * self_adj_corner + 12.5 * opponent_adj_corner

    
    def Mobility(self,game_state,agent_id):
        """
        Computed by finding the number of all possibe moves 
        """

        # get moves, -1 to exclude pass moves 
        ownMoves = len(self.gameRule.getLegalActions(game_state, agent_id))-1
        opponentMoves = len(self.gameRule.getLegalActions(game_state,(agent_id + 1)%2))-1

        moveDiff = ownMoves - opponentMoves
        totalMoves = ownMoves + opponentMoves

        if moveDiff == 0:
            return 0
        elif moveDiff > 0:
            return  100 * ownMoves / totalMoves
        else:
            return -100 * opponentMoves / totalMoves

            
    def validPos(self):
        pos_list = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                pos_list.append((x,y))
        return pos_list
    
    def frontierDiscs(self,game_state,agent_id):
        """
        Discs adjacent to empty squares. Most of these dics are volatile as they have a greater chance of being flipped by an 
        opponent's move in the empty square 
        -> Would like to minimize the number of frontier discs we have 
        """
        ownFrontiers = set()
        opFrontiers = set()

        # Compute frontier discs 
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                # own color 

                if game_state.board[x][y] == self.gameRule.agent_colors[agent_id]:
                    # enumerate over possible directions 
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        # valid + empty move 
                        #print(f"tem pos: {temp_pos}")
                        if temp_pos in self.validPos and game_state.getCell(temp_pos) == Cell.EMPTY:
                            ownFrontiers.add(pos) 

                # op frontier discs
                elif game_state.board[x][y] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                    # enumerate over possible directions 
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        #print(f"tem pos: {temp_pos}")
                        # valid + empty move 
                        if temp_pos in self.validPos and game_state.getCell(temp_pos) == Cell.EMPTY:
                            #print("hello 2?")
                            opFrontiers.add(pos)
        # compute metrics 
        ownFrontierLen = len(ownFrontiers)
        opFrontierLen = len(opFrontiers)
        lenDiff = ownFrontierLen - opFrontierLen
        totalFrontier = ownFrontierLen + opFrontierLen

        #print(f"debug ownFrontierLen: {ownFrontierLen}")
        #print(f"debug opFrontierLen: {opFrontierLen}")
        #print(f"debug lenDiff: {lenDiff}")

        # we don't want this... 
        if lenDiff > 0:
            return - 100 * ownFrontierLen / totalFrontier

        elif lenDiff < 0:
            return 100 * opFrontierLen / totalFrontier
            
        elif lenDiff == 0:
            return 0 


    def discSquares(self,game_state,agent_id):
        """ 
        """
        discSquareScore = 0 

        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                # self color 
                if game_state.board[i][j] == self.gameRule.agent_colors[agent_id]:
                    discSquareScore += self.DISC_SQUARES[i][j]
            
                # op color 
                elif game_state.board[i][j] == self.gameRule.agent_colors[(agent_id + 1)%2]:
                    discSquareScore -= self.DISC_SQUARES[i][j]
                # empty square -> do nothing


        return discSquareScore 
    
    def newEval(self,game_state,agent_id):
        if self.earlyGame:
            return  10 * self.pieceDifference(game_state,agent_id) + \
            801.724 * self.cornerOccupancy(game_state,agent_id) + \
            382.026 * self.cornerCloseness(game_state,agent_id) + \
            78.922 * self.Mobility(game_state,agent_id) + \
            20 * self.discSquares(game_state,agent_id) + \
            50.386 * self.frontierDiscs(game_state,agent_id)

        elif self.midGame:
            return  10 * self.pieceDifference(game_state,agent_id) + \
            801.724 * self.cornerOccupancy(game_state,agent_id) + \
            382.026 * self.cornerCloseness(game_state,agent_id) + \
            78.922 * self.Mobility(game_state,agent_id) + \
            10 * self.discSquares(game_state,agent_id) + \
            60 * self.frontierDiscs(game_state,agent_id)
            

        elif self.endGame:
            return 11 * self.pieceDifference(game_state,agent_id) + 801.724 * self.cornerOccupancy(game_state,agent_id) + \
            382.026 * self.cornerCloseness(game_state,agent_id) + \
            78.922 * self.Mobility(game_state,agent_id) + \
            10 * self.discSquares(game_state,agent_id) + \
            74.386 * self.frontierDiscs(game_state,agent_id)
