import math
import numpy as np
import time
from Reversi.reversi_model import boardToString


def ucb_score(parent, child):
    try:
        explore_score = math.sqrt(2*math.log(parent.visit_count) / (child.visit_count))
    except (ValueError, ZeroDivisionError):
        return 1000000

    
    if child.visit_count > 0:
        # Select child with smallest value for the opponent to maximize player's own score
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + explore_score


class Node:
    def __init__(self, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        
        action = actions[np.argmax(visit_counts)]

        return action

    def select_child(self):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_set):
        self.state = state
        for a in action_set:
            self.children[a] = Node((to_play + 1)%2)

class MCTS:

    def __init__(self, game, id, model = None):
        self.game = game
        #self.model = model
        self.id = id
        #self.count = 0
        self.STABILITY_WEIGHTS = [[4,  -1,  3,  3,  3,  3, -1,  4],
                                  [-1, -1,  0,  0,  0,  0, -1, -1],
                                  [3,   0,  0,  0,  0,  0,  0,  3],
                                  [3,   0,  0,  0,  0,  0,  0,  3],
                                  [3,   0,  0,  0,  0,  0,  0,  3],
                                  [3,   0,  0,  0,  0,  0,  0,  3],
                                  [-1, -1,  0,  0,  0,  0, -1, -1],
                                  [4,  -1,  3,  3,  3,  3, -1,  4]]
        self.corners_loc = [(0, 0), (0, 7), (7, 0), (7, 7)]
         
    def run(self, state):
        #self.count += 1
        #print("This is count MCTS: ", self.count)
        LIMIT = time.time() + 0.90
        #print("\nCount run: ", self.count)
        root = Node(self.id)
        valid_moves = list(set(self.game.getLegalActions(state, self.id)))
        originMoveLen = len(valid_moves)
        # test_next = self.game.generateSuccessor(state, valid_moves[0], self.id)
        # op_move = list(set(self.game.getLegalActions(test_next, (self.id + 1)%2)))
        # cornerFlag = False
        # for corner in self.corners_loc:
        #     if corner in op_move:
        #         print("This corner is now available to opponent: ", corner)
        #         print("This is the board state: ", boardToString(state.board, 8))
        #         cornerFlag = True

        #origin_moves = valid_moves
        root.expand(state, self.id, valid_moves)
        iter = 0
        while time.time() < LIMIT:
            # iter += 1
            # if cornerFlag and iter < 20:
            #     print("\nThis is iteration number: ", iter)
            #     for action, child in root.children.items():
            #         print("This is action and children value and visit count: ", action, child.value_sum, ucb_score(root, child), child.visit_count)

            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()

                # if cornerFlag and iter < 20:
                #     print("This is action taken with node value: ", action)

                search_path.append(node)

                # if cornerFlag and iter < 20:
                #     print("Len of search path now: ", len(search_path))

                # if self.count > 6:
                #     print("This is action selected: ", action)
                    
            parent = search_path[-2]
            state = parent.state
            next_state = self.game.generateSuccessor(state, action, parent.to_play)
            
            if self.game.getLegalActions(next_state,0) == ["Pass"] \
             and self.game.getLegalActions(state,1) == ["Pass"]:

                # Final score value at terminal state
                value = 1000 * (self.game.calScore(next_state, node.to_play) - self.game.calScore(next_state, parent.to_play))
            
            else:
                # Predict winners
                # value = model.predict(next_state)
                value = self.hPredict(next_state, node.to_play)
                valid_moves = list(set(self.game.getLegalActions(next_state, node.to_play)))
                # if cornerFlag and iter < 20:
                #     print("This is opponent moves: ", valid_moves)

                if iter <= originMoveLen:
                    for corner in self.corners_loc:
                        if corner in valid_moves:
                            value += 500
                        if corner == action:
                            value -= 500

                node.expand(next_state, node.to_play, valid_moves)
            
            # if value > 0:
            #     value = 1
            # if value <= 0:
            #     value = -1

            self.backpropagate(search_path, value, node.to_play)

        return root

    def backpropagate(self, search_path, value, to_play):

        #print("Value being propagated: ", value)
        #print("Search path len: ", len(search_path))
        discount = 0.9
        counter = 0
        for node in reversed(search_path):
            if node.to_play == to_play:
                #print("True")
                node.value_sum += (value * (discount**counter))
            else:
                #print("False")
                node.value_sum -= (value * (discount**counter))

            node.visit_count += 1
            counter += 1
    
    def hPredict(self, state, id):
        
        player =  self.game.calScore(state, id) + 10 * self.Heuristic(state, id)
        op = self.game.calScore(state, ((id + 1) % 2)) + 10 * self.Heuristic(state, ((id + 1) % 2))
        
        return player - op

    def getActualMobility(self,game_state,agent_id):
        ownmoves = 0 
        opponentmoves = 0 
        # get moves, -1 to exclude pass moves 
        ownmoves = len(list(set(self.game.getLegalActions(game_state, agent_id))))-1
        opponentmoves = len(list(set(self.game.getLegalActions(game_state,(agent_id + 1)%2))))-1

        if ownmoves + opponentmoves != 0:
            MobilityHeuristic = (float(ownmoves)-opponentmoves) / (float(ownmoves) +opponentmoves )

        else:
            MobilityHeuristic = 0 

        return MobilityHeuristic


    def getStability(self,game_state,agent_id):
        my_stability, tot = 0,0
        for i in range(8):
            for j in range(8):
                if game_state.board[i][j] == self.game.agent_colors[agent_id]:
                    my_stability += self.STABILITY_WEIGHTS[i][j]
                    tot += abs(self.STABILITY_WEIGHTS[i][j])
                elif game_state.board[i][j] == self.game.agent_colors[(agent_id + 1)%2]:
                    my_stability -= self.STABILITY_WEIGHTS[i][j]
                    tot += abs(self.STABILITY_WEIGHTS[i][j])

        if not tot == 0:
            return float(my_stability) / tot
        else:
            return 0

    def getCorners(self,game_state,agent_id):
        corners_loc = [(0,0), (0,7), (7,0), (7,7)]

        self_corner,opponent_corner = 0,0 

        for (i,j) in corners_loc: 
            if game_state.board[i][j] == self.game.agent_colors[agent_id]:
                self_corner += 1
            elif game_state.board[i][j] == self.game.agent_colors[(agent_id + 1)%2]:
                opponent_corner += 1 

        if float(self_corner) + opponent_corner != 0:
            return (float(self_corner) - opponent_corner) / (float(self_corner) + opponent_corner)
        else:
            return 0
        
    def Heuristic(self, game_state, agent_id):
        eval =  10 * self.getCorners(game_state,agent_id) + 3 * self.getActualMobility(game_state,agent_id) + \
            1 * self.getStability(game_state,agent_id)
        
        return eval