from collections import namedtuple
from random import choice
import math

from mcts import Node, MCTS

# move2ind = {}
# ind2move  = {}
# n = 0
# for i in range(9):
#     for j in range(i+1, 9):
#         move2ind[(i,j)] = n
#         move2ind[(j,i)] = n
#         ind2move[n] = (i,j)
#         n += 1
_Q3TB = namedtuple("Q3TBoard", "classic_state quantum_state isplayer1 winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Q3TBoard(_Q3TB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

def make_move(board:Q3TBoard, move:tuple[int, int]) -> Q3TBoard:
    qb = move2ind(*move)
    quantum_state = board.quantum_state[:qb] + (board.isplayer1,) + board.quantum_state[qb + 1:]
    print(quantum_state, qb)
    print(measure(move[0], None, quantum_state, visited=set()))
    return Q3TBoard(
        classic_state= board.classic_state,
        quantum_state= quantum_state,
        isplayer1= not board.isplayer1,
        winner= None,
        terminal= False
    )

def adjecent_states(s, q_state):
    # n1, n2 = ind2move(s)
    for sn in filter(lambda i: q_state[move2ind(i, s)] is not None, range(s)):
        yield sn
    for sn in filter(lambda i: q_state[move2ind(i, s)] is not None, range(s+1, 9)):
        yield sn
    # for sn in filter(lambda sn: q_state[sn] is not None and sn != s, map(lambda i: move2ind(n1, i), range(n1 + 1, 9))):
    #     yield sn
    # for sn in filter(lambda sn: q_state[sn] is not None and sn != s, map(lambda i: move2ind(i, n2), range(n2))):
    #     yield sn
    # for sn in filter(lambda sn: q_state[sn] is not None and sn != s, map(lambda i: move2ind(i, n2), range(n2 + 1, 9))):
    #     yield sn

    

def measure(s, parent, q_state, visited:set=set()):
    # This DFS finds a cycle in an undirected graph
    if s in visited:
        return True
    visited.add(s)
    for sn in adjecent_states(s, q_state):
        if sn not in visited:
            if measure(sn, s, q_state, visited):
                return True
        elif sn != parent:
            return True
    return False


def move2ind(i, j):
    if i > j:
        return move2ind(j, i)
    # return 8*i - (i*i - i)//2
    # return  8*i - (i*i - i)//2 + j - i - 1
    return  (15*i - i*i + 2*j - 2)//2

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j

def new_q3t_board() -> Q3TBoard:
    return Q3TBoard(
        classic_state=(None,)*9,
        quantum_state=(None,)*36, # comb(9, 2) = 36
        isplayer1=True,
        winner=None,
        terminal=False)

def play_game():
    # tree = MCTS()
    board = new_q3t_board()
    move = (0, 2)
    # print(board.quantum_state)
    board = make_move(board, move)
    move = (0, 4)
    board = make_move(board, move)
    move = (3, 4)
    board = make_move(board, move)
    move = (3, 0)
    board = make_move(board, move)
    # print(board)
    quit()


if __name__ == "__main__":
    # print(move2ind(1,2))
    # print(move2ind(0,1))
    # print(move2ind(1,2))
    # print(ind2move(34))
    # print(ind2move)
    play_game()