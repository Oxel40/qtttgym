from collections import namedtuple
from random import choice
import math
import random

from mcts import Node, MCTS
import qtttgym

# move2ind = {}
# ind2move  = {}
# n = 0
# for i in range(9):
#     for j in range(i+1, 9):
#         move2ind[(i,j)] = n
#         move2ind[(j,i)] = n
#         ind2move[n] = (i,j)
#         n += 1
# _Q3TB = namedtuple("Q3TBoard", "classic_state quantum_state isplayer1 winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Q3TBoard(qtttgym.Board, Node):
    def __init__(self):
        qtttgym.Board.__init__(self, qtttgym.QEvalClassic())
        Node.__init__(self)
        self.terminal = False
        self.winner = None

    def turn(self):
        return len(self.moves) % 2

    def find_children(self):
        if self.terminal:
            return set()
        out = set()
        for i in range(36):
            move = ind2move(i)
            if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                continue
            out.add(self.step(move))
        return out

    def find_random_child(self):
        if self.terminal:
            return None
        while True: 
            i = random.randint(0, 35)
            move = ind2move(i)
            if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                continue
            return self.step(move)

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        
        if self.winner is self.turn():
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self}")
        if self.turn() is (not self.winner):
            return -1  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def is_terminal(self):
        return self.terminal
    
    def step(self, move: tuple[int, int]):
        new_state = Q3TBoard()
        new_state.make_move(move)
        p1, p2 = new_state.check_win()
        if p1 > p2 and p1 > 0:
            # p1 is the winner
            new_state.winner = 0
            new_state.terminal = True
        elif p2 > p1 and p2 > 0:
            # p2 is the winner
            new_state.winner = 1
            new_state.terminal = True
        new_state.terminal = len(new_state.moves) == 9 or new_state.terminal
        return new_state
    
    def __eq__(node1, node2) -> bool:
        return hash(node1) == hash(node2)
    
    def __hash__(self) -> int:
        return self.hash()

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j

def play_game():
    tree = MCTS()
    board = Q3TBoard()
    while True:
        for i in range(10):
            tree.do_rollout(board)
            print(i)
        board = tree.choose(board)
        qtttgym.displayBoard(board)
        quit()


if __name__ == "__main__":
    # print(move2ind(1,2))
    # print(move2ind(0,1))
    # print(move2ind(1,2))
    # print(ind2move(34))
    # print(ind2move)
    play_game()