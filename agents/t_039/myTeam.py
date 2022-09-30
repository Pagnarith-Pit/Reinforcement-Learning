from template import Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import math
import random
import time

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
    
    def selection(self, parent_state, game_state_child_list, visited_state_dict):
        maxUCB = 0
        selected = game_state_child_list[0]
        parent_state_rep = boardToString(parent_state.board, self.GRID_SIZE)
        parent_num_of_visit = visited_state_dict[parent_state_rep][1]

        for state in game_state_child_list:
            state_rep = boardToString(state.board, self.GRID_SIZE)
            if state_rep not in visited_state_dict:
                UCB_val = float('inf')
                selected = state
                return selected
          
            total_score, num_of_visit = visited_state_dict[state_rep]

            # try:
            UCB_val = (total_score/num_of_visit) + math.sqrt(2* math.log(parent_num_of_visit)/num_of_visit)
            # except (ZeroDivisionError):
            #     for key, val in visited_state_dict.items():
            #         print("This is the key in visited_state_dict: ", key)
            #         print("This is its value: ", val)

            #         print("This state raised Zero Div Error: ", state_rep)
            #         exit()

            if UCB_val > maxUCB:
                maxUCB = UCB_val
                selected = state

        return selected

    def expansion(self, game_state, actions, move_id):

        actions = list(set(actions))
        game_state_child_list = []
        for action in actions:
            childState = self.gameRule.generateSuccessor(game_state, action, move_id)
            game_state_child_list.append(childState)
        
        return game_state_child_list

    def simulation(self, game_state, op_id, player_id):
        total_score_final = 0

        opponent_id = op_id
        opponent_actions = list(set(self.gameRule.getLegalActions(game_state, opponent_id)))
        rand_move_op = random.choice(opponent_actions)
        game_state = self.gameRule.generateSuccessor(game_state, rand_move_op, opponent_id)

        player_actions = list(set(self.gameRule.getLegalActions(game_state, player_id)))

        while opponent_actions != ["Pass"] and player_actions != ["Pass"]:
            rand_move_player  = random.choice(player_actions)
            game_state = self.gameRule.generateSuccessor(game_state, rand_move_player, player_id)

            opponent_actions = list(set(self.gameRule.getLegalActions(game_state, opponent_id)))
            rand_move_op = random.choice(opponent_actions)
            game_state = self.gameRule.generateSuccessor(game_state, rand_move_op, opponent_id)

            player_actions = list(set(self.gameRule.getLegalActions(game_state, player_id)))

        total_score_final = self.gameRule.calScore(game_state, self.id) - self.gameRule.calScore(game_state, (self.id + 1)%2)
       
        return total_score_final

    def backPropogation(self, total_score_final, selection_list, visited_state_dict, bonus):

        for state in selection_list:
            state_rep = boardToString(state.board, self.GRID_SIZE)
            if state_rep in visited_state_dict:
                total_score, num_of_visit = visited_state_dict[state_rep]
                total_score = total_score + total_score_final + bonus
                visited_state_dict[state_rep] = (total_score, num_of_visit + 1)
            else:
                visited_state_dict[state_rep] = (total_score_final + bonus, 1)
    
    def endingState(self, game_state):
        if self.gameRule.getLegalActions(game_state,0) == ["Pass"] \
            and self.gameRule.getLegalActions(game_state,1) == ["Pass"]:
            return True
        else: return False

    # This expresses domain knowledge
    def calculateBonusScore(self, action):
        STABILITY_WEIGHTS = [[4,  -3,  2,  2,  2,  2, -3,  4],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [2,  -1,  0,  1,  1,  0, -1,  2],
                   [2,  -1,  0,  1,  1,  0, -1,  2],
                   [2,  -1,  1,  0,  0,  1, -1,  2],
                   [-3, -4, -1, -1, -1, -1, -4, -3],
                   [4,  -3,  2,  2,  2,  2, -3,  4]]
        y, x = action
        bonus = STABILITY_WEIGHTS[y][x]

        return bonus

    def SelectAction(self,actions,game_state):
        TIME_LIMIT = time.time() + 0.96
        self.GRID_SIZE = game_state.grid_size
        visited_state_dict = dict()
        actions = list(set(actions))

        visited_state_dict[boardToString(game_state.board, self.GRID_SIZE)] = (0, 0)
        player_id = self.id
        opponent_id = (player_id + 1)%2
        self.gameRule.agent_colors = game_state.agent_colors
        parent_state_init = game_state
        agent_turn = [player_id, opponent_id]

        if actions == ["Pass"]:
            return "Pass"

        while time.time() < TIME_LIMIT:
            gameEndFlag = False
            selection_list = [parent_state_init]
            game_state_child_list = self.expansion(parent_state_init, actions, player_id)
            best_child_state = self.selection(parent_state_init, game_state_child_list, visited_state_dict)
            selection_list.append(best_child_state)
            index = game_state_child_list.index(best_child_state)
            best_action = actions[index]
            

            #Add bonus
            bonus = self.calculateBonusScore(best_action)
            
            i = 1
            best_child_state_rep = boardToString(best_child_state.board, self.GRID_SIZE)
            while best_child_state_rep in visited_state_dict:
          
                parent_state_next = best_child_state
                actions_next = list(set(self.gameRule.getLegalActions(parent_state_next, agent_turn[i])))
             
                if self.endingState(parent_state_next):
                    total_score_final = self.gameRule.calScore(game_state, player_id) - self.gameRule.calScore(game_state, opponent_id) 
                    self.backPropogation(total_score_final, selection_list, visited_state_dict, bonus)
                    gameEndFlag = True
                    break

                game_state_child_list = self.expansion(parent_state_next, actions_next, agent_turn[i])
                best_child_state = self.selection(parent_state_next, game_state_child_list, visited_state_dict)

                best_child_state_rep = boardToString(best_child_state.board, GRID_SIZE)

                # for passing, this will append child state twice
                selection_list.append(best_child_state)

                # if child is ending state, stop here, return total score, propgate, and continue
                i = (i + 1)%2

            if gameEndFlag:
                continue

            total_score_final = self.simulation(best_child_state, agent_turn[i], agent_turn[(i + 1)%2])
            self.backPropogation(total_score_final, selection_list, visited_state_dict, bonus)
        print("This is the best action: ", best_action)
        try:
            return best_action
        except:
            print("This is the error")