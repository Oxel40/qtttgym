from collections import namedtuple
from random import choice
import math
import random

from mcts import Node, MCTS
import qtttgym

from copy import deepcopy
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
        r1, r2 = self.check_win()
        if r1 < r2 and r1 > 0:
            # player 1 is the winner
            self.winner = 0
        elif r2 < r1 and r2 > 0:
            self.winner = 1
        if self.winner == self.turn():
            return 1
        elif self.winner == None:
            return 0
        return -1
        
    def is_terminal(self):
        return self.terminal
    
    def step(self, move: tuple[int, int]):
        new_state = deepcopy(self)
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

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j

def play_game():
    tree = MCTS()
    board = Q3TBoard()
    terminal = False
    qtttgym.displayBoard(board)
    while not terminal:
        move = tuple(map(int, input("make a move: ").split()))
        board = board.step(move)
        qtttgym.displayBoard(board)
        terminal = board.terminal
        if terminal: break

        for i in range(2000):
            tree.do_rollout(board)
            # print(i)
        board = tree.choose(board)
        qtttgym.displayBoard(board)
        terminal = board.terminal
    print(board.winner)
    print()

if __name__ == "__main__":
    # print(move2ind(1,2))
    # print(move2ind(0,1))
    # print(move2ind(1,2))
    # print(ind2move(34))
    # print(ind2move)
    play_game()