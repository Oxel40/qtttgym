"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""

from collections import namedtuple
from random import choice
from mcts import MCTS, Node
import numpy as np
import math
from tqdm import trange
import qtttgym
import random
from copy import deepcopy

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

NodeType = int 

class QTTTGame():
    class GameState(qtttgym.Board):
        
        def __init__(self, board, moves, turn, winner, terminal):
            qtttgym.Board.__init__(self, qtttgym.QEvalClassic())
            # State params
            self.board = board
            self.moves:list = moves
            self.turn = turn
            self.winner = winner
            self.terminal = terminal
            # necessary stuff
            
            # actions
            self.actions = list()
            self.children:dict[int, list] = dict()
            for i in range(36):
                move = ind2move(i)
                if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                    continue
                self.actions.append(i)
                self.children[i] = None

            # MCTS properties
            self.N_tot = 0
            self.N:dict = {a : 0 for a in self.actions}
            self.W:dict = {a : 0 for a in self.actions}
            self.Q:dict = {a : 0 for a in self.actions}
            self.P:dict | None = None
        
        def update_actions(self):
            self.actions = list()
            for i in range(36):
                move = ind2move(i)
                if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                    continue
                self.actions.append(i)
                self.children[i] = None
            self.N:dict = {a : 0 for a in self.actions}
            self.W:dict = {a : 0 for a in self.actions}
            self.Q:dict = {a : 0 for a in self.actions}
            self.P:dict | None = None

        def update_winner(self):
            p1, p2 = self.check_win()
            if p1 > 0 and p2 > 0:
                self.winner = p1 < p2
                self.terminal = True
            elif p2 < 0 and p1 > 0:
                # p1 is the winner
                self.winner = True
                self.terminal = True
            elif p1 < 0 and p2 > 0:
                # p2 is the winner
                self.winner = False
                self.terminal = True
            self.terminal = len(self.moves) == 9 or self.terminal

        def __hash__(self) -> int:
            return hash(tuple(self.board) + tuple(self.moves))
        
        def __str__(self) -> str:
            list_of_buffers = [[' '] * 9 for _ in range(9)]

            for i, m in enumerate(self.moves):
                list_of_buffers[m[0]][i] = str(i)
                list_of_buffers[m[1]][i] = str(i)

            for i, b in enumerate(self.board):
                if b >= 0:
                    for j in range(9):
                        if (j % 2 == 0 and b % 2 == 0):
                            list_of_buffers[i][j] = 'x'
                        elif (j % 2 == 1 and b % 2 == 1):
                            list_of_buffers[i][j] = 'o'
                        else:
                            list_of_buffers[i][j] = ' '
                    list_of_buffers[i][4] = str(b)

            out_string = ""
            for i in range(3):
                out_string += "+---+---+---+\n"
                for k in range(3):
                    for j in range(3):
                        out_string += "|"
                        out_string += "".join(list_of_buffers[i *
                                            3 + j][k * 3: k * 3 + 3])
                    out_string += "|\n"
            out_string += "+---+---+---+\n"
            return out_string
        
        def __repr__(self) -> str:
            return f"GameState({self.board},{self.P is not None})"

    def __init__(self) -> None:
        self.root = self.GameState([-1]*9, [], True, None, False)
        self.c_puct = 1
        self.nodes:dict[int, self.GameState] = dict()
        self.nodes[hash(self.root)] = self.root

    def make_move(self, action):
        if action not in self.root.children:
            raise Exception("Invalid Action")
        if self.root.children[action] is None:
            # node has not been fully expanded
            # only expand the child of this action
            self._expand_child(self.root, action)
        self.root = np.random.choice(self.root.children[action])

    def choose(self):
        n = self.root
        def score(a):
            # if n.turn:
            if n.N[a] == 0:
                return -math.inf
            return n.Q[a]
            # else:
            #     if n.N[a] == 0:
            #         return math.inf
            #     return -n.Q[a]

        # print()
        # print([score(a) for a in self.root.actions])
        a_best = max(self.root.actions, key=score)
        # print(a_best)
        return a_best
    
    def _expand_child(self, node:GameState, action):
        nodes = self._step(node, action)
        node.children[action] = nodes
        
    def _step(self, node:GameState, action) -> list[GameState]:
        move = ind2move(action)
        new_node = self.GameState(node.board.copy(), 
                                  node.moves.copy(), 
                                  node.turn, 
                                  None, 
                                  False)
        # new_node = deepcopy(node)
        new_node.qstructs = deepcopy(node.qstructs)
        new_node.make_move(move)
        new_node.turn = not new_node.turn
        new_node.update_winner()
        if node.board == new_node.board:
            return [new_node]
        # A State Collapse has happened
        hashes = {hash(new_node)}
        new_node.update_actions()
        out = [new_node]

        while len(hashes) < 2:
            new_node = self.GameState(node.board.copy(), 
                                      node.moves.copy(), 
                                      node.turn, 
                                      None, 
                                      False)
            new_node.qstructs = deepcopy(node.qstructs)
            new_node.make_move(move)

            hashes.add(hash(new_node))
            new_node.update_actions()
        
        new_node.turn = not new_node.turn
        new_node.update_winner()
        out.append(new_node)
        return out

    def do_rollout(self):
        "Make the tree one layer better. (Train for one iteration.)"
        # select until we uct select a new node
        path, leaf = self._select(self.root)
        r_tot = 0
        N = 10
        for _ in range(N):
            r = self._simulate(leaf)
            r_tot += r if leaf.turn else -r
        # input()
        path.append((leaf, None))
        # for s, _ in path:
        #     print(s)
        # backprop
        self._backpropogate(path, r_tot/N)
        
    
    def _select(self, node:GameState) -> tuple[list[GameState], GameState]:
        path = []
        while (node.P is not None) and not node.terminal:
            a = self._uct_select(node)
            if node.children[a] is None:
                self._expand_child(node, a)
            path.append((node, a))
            node = np.random.choice(node.children[a])
        return path, node

    def _simulate(self, node:GameState) -> list[tuple[GameState, int]]:
        # Simulates from the node to the bottom
        invert_reward = True
        while not node.terminal:
            if node.P is None:
                # Evaluate the action probabilities of this Node
                node.P = self.get_action_probs(node)
            # Sample an action according to P
            a = self.sample_action(node)
            nodes = self._step(node, a)
            node = np.random.choice(nodes)
            # node = node.children[a]
            # node = self._step(node, a)
            invert_reward = not invert_reward
        # print(node)
        # print(node.board)
        # print(node.check_win())
        # print(node.winner)
        r = self._reward(node)
        # if invert_reward:
        #     r = -r
        # print(r)
        # input()
        # print("END")
        # print(node)
        # print(node.winner, node.turn, invert_reward)
        return r

    def _backpropogate(self, path:list[tuple[GameState, int]], r):
        leaf, _ = path.pop()
        while path:
            node, a = path.pop()
            # print(node)
            r = -r
            node.W[a] += r
            node.N[a] += 1
            node.Q[a] = node.W[a] / node.N[a]
            node.N_tot += 1
            # print(node.Q)
            # input()
        
    def _reward(self, node:GameState):
        r = 0
        if node.winner is None:
            return 0
        if node.winner:
            r = 1
        else:
            r = -1
        # if len(node.moves)
        return r
        
        
    
    def _uct_select(self, node:GameState):
        # Return the selected action
        def uct(a):
            # if node.turn:
            U = node.P[a] * math.sqrt(node.N_tot)/(1 + node.N[a]) 
            # if node.turn:
            return node.Q[a] + self.c_puct * U
            # else:
            #     return -node.Q[a] + self.c_puct * U
        # print(node)
        # for a in node.actions:
        #     print(a, uct(a))
        # input()
        return max(node.actions, key=uct)

    def get_action_probs(self, node:GameState) -> dict:
        # This should be replaced by a policy NN
        return {a : 1/len(node.actions) for a in node.actions}
    
    def sample_action(self, node:GameState):
        return np.random.choice(node.actions, p=list(node.P.values()))
        
    
def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal

def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j

if __name__ == "__main__":
    # play_game()
    # np.random.seed(1)
    # random.seed(1)
    
    game = QTTTGame()
    print(game.root)
    # quit()
    while not game.root.terminal:
        # print(game.root)
        rollout_bar = trange(30000, ncols=150)
        for i in rollout_bar:
            game.do_rollout()
            node = game.root
            qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)

            q = " ".join([f"{len(game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
            rollout_bar.set_description_str(f"{q}")
        a = game.choose()
        game.make_move(a)
        print(game.root)
        a = map(int, input("Make move: ").split())
        game.make_move(a)
        # input()
    print(game.root.winner)