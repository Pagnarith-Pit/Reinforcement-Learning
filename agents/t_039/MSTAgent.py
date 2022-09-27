from template import Agent
from Reversi.reversi_model import *
from Reversi.reversi_utils import boardToString
import math
import random

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = ReversiGameRule(2)
    
    def selection(self, parent_state, game_state_child_list, visited_state_dict):
        maxUCB = 0
        selected = game_state_child_list[0]
        parent_num_of_visit = visited_state_dict[parent_state][1]

        for state in game_state_child_list:
            if state not in visited_state_dict:
                UCB_val = float('inf')
                selected = state
                return selected
          
            total_score, num_of_visit = visited_state_dict[state]
            UCB_val = (total_score/num_of_visit) + 2*math.sqrt(math.log(parent_num_of_visit)/num_of_visit)

            if UCB_val > maxUCB:
                maxUCB = UCB_val
                selected = state
        
        return selected

    def expansion(self, game_state, actions, move_id):

        game_state_child_list = []
        for action in actions:
            childState = self.gameRule.generateSuccessor(game_state, action, move_id)
            game_state_child_list.append(childState)
        
        return game_state_child_list

    def simulation(self, game_state, op_id, player_id):
        total_score_final = 0

        opponent_id = op_id
        opponent_actions = self.gameRule.getLegalActions(game_state, opponent_id)
        rand_move_op = random.choice(opponent_actions)
        game_state = self.gameRule.generateSuccessor(game_state, rand_move_op, opponent_id)

        player_actions = self.gameRule.getLegalActions(game_state, player_id)

        while opponent_actions != ["Pass"] and player_actions != ["Pass"]:
            rand_move_player  = random.choice(player_actions)
            game_state = self.gameRule.generateSuccessor(game_state, rand_move_player, player_id)

            opponent_actions = self.gameRule.getLegalActions(game_state, opponent_id)
            rand_move_op = random.choice(opponent_actions)
            game_state = self.gameRule.generateSuccessor(game_state, rand_move_op, opponent_id)

            player_actions = self.gameRule.getLegalActions(game_state, player_id)


        total_score_final = self.gameRule.calScore(game_state, self.id) - self.gameRule.calScore(game_state, self.gameRule.getNextAgentIndex())

        return total_score_final

    def backPropogation(self, total_score_final, selection_list, visited_state_dict):

        for state in selection_list:
            if state in visited_state_dict:
                total_score, num_of_visit = visited_state_dict[state]
                total_score += total_score_final
                visited_state_dict[state] = (total_score, num_of_visit + 1)
            else:
                visited_state_dict[state] = (total_score_final, 1)
        
    def SelectAction(self,actions,game_state):
        ITERATION = 20
        visited_state_dict = dict()

        visited_state_dict[game_state] = (0, 0)
        player_id = self.id
        opponent_id = (player_id + 1)%2
        self.gameRule.agent_colors = game_state.agent_colors
        parent_state_init = game_state
        agent_turn = [player_id, opponent_id]

        print("Initial board state: ", boardToString(parent_state_init.board, 8))
        print("Initial Actions is: ", actions)
        print("This is player_id" , player_id)

        # Testing to see child state
        # game_state_child_list = self.expansion(parent_state_init, actions, player_id)

        # print("Testing")
        # game_state_child_list = self.expansion(parent_state_init, actions, player_id)

        # print("This is the child state")
        # for state in game_state_child_list:
        #     print(boardToString(state.board, 8))

    #########################

        breakFlag = False
        if actions == ["Pass"]:
            passCount += 1

        for time in range(ITERATION):
            
            selection_list = [parent_state_init]
            print("This is the num of iter: " , time)
            
            game_state_child_list = self.expansion(parent_state_init, actions, player_id)
            best_child_state = self.selection(parent_state_init, game_state_child_list, visited_state_dict)
            selection_list.append(best_child_state)
            
            i = 1
            while best_child_state in visited_state_dict:
                parent_state_next = best_child_state
                actions_next = self.gameRule.getLegalActions(parent_state_next, agent_turn[i])
                if actions_next == ["Pass"]:
                    passCount += 1
                else:
                    passCount = 0
                
                if passCount == 2:
                    total_score_final = self.gameRule.calScore(game_state, player_id) - self.gameRule.calScore(game_state, opponent_id) 
                    self.backPropogation(total_score_final, selection_list)
                    breakFlag = True
                    break

                game_state_child_list = self.expansion(parent_state_next, actions_next, agent_turn[i])
                best_child_state = self.selection(parent_state_next, game_state_child_list)

                # for passing, this will append child state twice
                selection_list.append(best_child_state)

                # if child is ending state, stop here, return total score, propgate, and continue
                i = (i + 1)%2

            if breakFlag:
                continue

            total_score_final = self.simulation(best_child_state, agent_turn[i], agent_turn[(i + 1)%2])
            self.backPropogation(total_score_final, selection_list, visited_state_dict)


        ### Need to fix how states get added to visited. The number must increase during backpropagation
        # Return best move here
        return random.choice(actions)