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

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")


class TTTGame():
    class GameState():
        
        def __init__(self, board, turn, winner, terminal):
            # State params
            self.board = board
            self.turn = turn
            self.winner = winner
            self.terminal = terminal
            
            # actions
            self.actions = list()
            self.children = dict()
            for i in range(9):
                if self.board[i] is not None: continue
                self.actions.append(i)
                self.children[i] = None
                    
            
            # MCTS properties
            self.N_tot = 0
            self.N:dict = {a : 0 for a in self.actions}
            self.W:dict = {a : 0 for a in self.actions}
            self.Q:dict = {a : 0 for a in self.actions}
            self.P:dict|None = None


        def __hash__(self) -> int:
            return hash(self.board + self.turn + self.winner + self.terminal)
        
        def __str__(self) -> str:
            to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
            rows = [
                [to_char(self.board[3 * row + col]) for col in range(3)] for row in range(3)
            ]
            return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )
        def __repr__(self) -> str:
            return f"GameState({self.board},{self.P is not None})"

    def __init__(self) -> None:
        self.root = self.GameState((None,)*9, True, None, False)
        self.c_puct = 10

    def make_move(self, action):
        if action not in self.root.children:
            raise Exception("Invalid Action")
        if self.root.children[action] is None:
            # node has not been fully expanded
            # only expand the child of this action
            self._expand_child(self.root, action)
        self.root = self.root.children[action]

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
    
    def _expand(self, node:GameState):
        for a in node.actions:
            if a in node.children: continue
            self._expand_child(node, a)
    
    def _expand_child(self, node:GameState, action):
        node.children[action] = self._step(node, action)

    def _step(self, node:GameState, action):
        new_board = node.board[:action] + (node.turn,) + node.board[action + 1 :]
        turn = not node.turn
        winner = _find_winner(new_board)
        is_terminal = (winner is not None) or not any(v is None for v in new_board)
        return self.GameState(new_board, turn, winner, is_terminal)

    def do_rollout(self):
        "Make the tree one layer better. (Train for one iteration.)"
        # select until we uct select a new node
        path, leaf = self._select(self.root)
        # print(path)
        # input()
        # simulate from the new leaf to the bottom
        # print()
        # print("Start")
        # print(leaf)
        reward = self._simulate(leaf)
        # print(reward)
        # input()
        path.append((leaf, None))
        # for s, _ in path:
        #     print(s)
        # backprop
        self._backpropogate(path, reward)
        
    
    def _select(self, node:GameState) -> tuple[list[GameState], GameState]:
        path = []
        while (node.P is not None) and not node.terminal:
            a = self._uct_select(node)
            if node.children[a] is None:
                self._expand_child(node, a)
            path.append((node, a))
            node = node.children[a]
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
            node = self._step(node, a)
            invert_reward = not invert_reward
        r = self._reward(node)
        if invert_reward:
            r = -r
        # print("END")
        # print(node)
        # print(node.winner, node.turn, invert_reward)
        return r

    def _backpropogate(self, path:list[tuple[GameState, int]], r):
        leaf, _ = path.pop()
        while path:
            node, a = path.pop()
            node.W[a] += r
            node.N[a] += 1
            node.Q[a] = node.W[a] / node.N[a]
            node.N_tot += 1
            r = -r
        #     print(r)
        # input()
            
        
    def _reward(self, node:GameState):
        if node.winner is None:
            return 0
        if node.turn is (not node.winner):
            return -1
        
    
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

if __name__ == "__main__":
    # play_game()
    game = TTTGame()
    # game.make_move(4)
    # game.make_move(6)
    # game.make_move(3)
    # game.make_move(5)
    # game.make_move(0)
    # game.make_move(8)
    # game.make_move(7)
    print(game.root)
    while not game.root.terminal:
        a = int(input("Make move: "))
        game.make_move(a)
        print(game.root)
        rollout_bar = trange(50000)
        for i in rollout_bar:
            game.do_rollout()
            node = game.root
            c_expl = [node.N[a] for a in node.actions]
            c_expl = " ".join([f"{x}" for x in c_expl])
            q = " ".join([f"{x:.3f}" for x in node.Q.values()])
            rollout_bar.set_description_str(f"{q} | {c_expl}")
            # input()
        a = game.choose()
        game.make_move(a)
        print(game.root)
    