from abc import abstractclassmethod
import time
from typing import Callable
import qtttgym
import math
from copy import deepcopy
import numpy as np


class Strategy:
    def __init__(self) -> None:
        ...

    def reset(self, game):
        """reset the game to the starting state
        """
        self.game = game
    
    @abstractclassmethod
    def contemplate(self, thinking_time):
        """spend some time planning your move

        Args:
            time (int): time limit 
        """

    @abstractclassmethod
    def choose(self):
        """make a choice
        Return:
            a (int): action
        """
        ...
    
    @abstractclassmethod
    def sync(self, a):
        """Sync your strategy given the move that was just made

        Args:
            a (int): action that was taken by your opponent
            sn (_type_): next state that was transitioned too
        """
        ...


class MCTS(Strategy):
    class GameState(qtttgym.Board):
        def __init__(self, board, moves, turn, winner, terminal):
            qtttgym.Board.__init__(self, qtttgym.QEvalClassic())
            # State params
            self.board = board
            self.moves:list = moves
            self.turn = turn
            self.winner = winner
            self.terminal = terminal

            # actions
            self.actions = list()
            self.children = dict()
            for a in range(36):
                move = ind2move(a)
                if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                    continue
                self.actions.append(a)
                self.children[a] = None
            
            # MCTS properties
            self.Ntot = 0
            self.N:dict = {a:0 for a in self.actions}
            self.W:dict = {a:0. for a in self.actions}
            self.Q:dict = {a:0. for a in self.actions}
            self.P:dict | None = None

            # details
            self.ref_count = 0

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

        def to_vector(self):
            classic_state = np.zeros((9, 10))
            for i in range(9):
                classic_state[i][self.board[i]] = 1
            quantum_state = np.zeros((9, 10))
            isqrt2 = 1/math.sqrt(9)
            for (i, j, t) in self.moves:
                quantum_state[i, t] = isqrt2
                quantum_state[j, t] = isqrt2

            qsets = set()
            for s in self.qstructs:
                qsets = qsets.union(s)
            
            for s in range(9):
                if s not in qsets:
                    quantum_state[s, -1] = 1.
                    continue
            return np.concatenate((classic_state, quantum_state), axis=0)
        
        def action_mask(self):
            a = np.zeros(36).astype(bool)
            for i in self.actions:
                a[i] = True
            return a
        
        def __hash__(self) -> int:
            return hash(tuple(self.board) + tuple(self.moves))
        
        def __eq__(self, other) -> bool:
            return hash(self) == hash(other)

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
            return f"GameState({self.board},{self.winner})"
    
    def __init__(self, rollouts:int=5000) -> None:
        super().__init__()
        self.c_puct = 1.
        self.nodes:dict[int, self.GameState] = dict()
        self.num_rollouts = rollouts

    def reset(self, game:qtttgym.Board):
        super().reset(game)
        turn = len(game.moves) % 2 == 0
        w1, w2 = game.check_win()
        terminal = False
        winner = None
        if w1 > 0 and w2 > 0:
            winner = w1 < w2
            terminal = True
        elif w2 < 0 and w1 > 0:
            # p1 is the winner
            winner = True
            terminal = True
        elif w1 < 0 and w2 > 0:
            # p2 is the winner
            winner = False
            terminal = True
        terminal = len(game.moves) == 9 or terminal
        self.root = self.GameState(
            game.board.copy(), 
            game.moves.copy(), 
            turn, 
            winner, 
            terminal)
        self.nodes:dict[int, self.GameState] = dict()
        self.nodes[hash(self.root)] = self.root
    
    def _rollout(self, num_sims:int=10):
        path, leaf = self._select(self.root)
        r_tot = 0
        for _ in range(num_sims):
            r = self._simulate(leaf)
            r_tot += r if leaf.turn else -r
        path.append((leaf, None))
        self._backpropogate(path, r_tot/num_sims)
    
    def _backpropogate(self, path:list[tuple[GameState, int]], r):
        leaf, _ = path.pop()
        while path:
            node, a = path.pop()
            r = -r
            node.W[a] += r
            node.N[a] += 1
            node.Q[a] = node.W[a] / node.N[a]
            node.Ntot += 1

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
            invert_reward = not invert_reward
        r = self._reward(node)
        return r
    
    def _reward(self, node:GameState):
        r = 0
        if node.winner is None:
            return 0
        if node.winner:
            r = 1
        else:
            r = -1
        return r
    
    def _expand_child(self, node:GameState, action):
        new_nodes = self._step(node, action)
        node.children[action] = []
        for n in new_nodes:
            n_hash = hash(n)
            if n_hash in self.nodes:
                n = self.nodes[n_hash]
            else:
                self.nodes[n_hash] = n
            node.children[action].append(n)
            n.ref_count += 1
    
    def _prune(self, node:GameState):
        if node is None:
            return
        node.ref_count -= 1
        if node.ref_count > 0: return
        del self.nodes[hash(node)]
        for a in node.actions:
            if node.children[a] is None: continue
            for child in node.children[a]:
                self._prune(child)
        
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
    
    def _select(self, node:GameState):
        path = []
        while (node.P is not None) and not node.terminal:
            a = self._uct_select(node)
            if node.children[a] is None:
                self._expand_child(node, a)
            path.append((node, a))
            node = np.random.choice(node.children[a])
        return path, node
        ...
        
    def _uct_select(self, node:GameState):
        # Return the selected action
        def uct(a):
            U = node.P[a] * math.sqrt(node.Ntot)/(1 + node.N[a])
            return node.Q[a] + self.c_puct * U
        return max(node.actions, key=uct)

    def get_action_probs(self, node:GameState) -> dict:
        # This should be replaced by a policy NN
        return {a : 1/len(node.actions) for a in node.actions}
    
    def sample_action(self, node:GameState):
        return np.random.choice(node.actions, p=list(node.P.values()))

    def contemplate(self, thinking_time, debug_info=False):
        t0 = time.time()
        n = 0
        while time.time() - t0 < thinking_time and n < self.num_rollouts:
            self._rollout(num_sims=10)
            qvals = sorted(list(self.root.Q.items()), key=lambda x: x[1], reverse=True)
            q = " ".join([f"{len(self.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
            q += f" | {self.root.Ntot}"
            if debug_info:
                print(q, end="\r")
            n += 1
        if debug_info:
            print()

    def choose(self):
        n = self.root
        def score(a):
            if n.N[a] == 0:
                return -math.inf
            return n.Q[a]
        a_best = max(self.root.actions, key=score)
        return a_best

    def sync(self, action):
        if action not in self.root.children:
            raise Exception("Invalid Action")
        if self.root.children[action] is None:
            # node has not been fully expanded
            # only expand the child of this action
            self._expand_child(self.root, action)

        options:list[self.GameState] = self.root.children[action]
        hashes = [hash(v) for v in options]
        game_hash = self.GameState.__hash__(self.game)
        i = hashes.index(game_hash)
        new_root:self.GameState = options[i]
        for action in self.root.actions:
            if self.root.children[action] is None: continue
            for child in self.root.children[action]:
                if child == new_root: continue
                self._prune(child)
        # The root should have no reference counter
        del self.nodes[hash(self.root)]
        self.root = new_root

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j

def move2ind(i, j):
    if i > j:
        return move2ind(j, i)
    # return 8*i - (i*i - i)//2
    # return  8*i - (i*i - i)//2 + j - i - 1
    return  (15*i - i*i + 2*j - 2)//2

def check_win(game:qtttgym.Board):
    p1, p2 = game.check_win()
    winner = None
    if p1 > 0 and p2 > 0:
        winner = p1 < p2
    elif p2 < 0 and p1 > 0:
        # p1 is the winner
        winner = True
    elif p1 < 0 and p2 > 0:
        # p2 is the winner
        winner = False
    return winner

def play_game(player1:Strategy, player2:Strategy, thinking_time:int=10):
    game = qtttgym.Board(qtttgym.QEvalClassic())
    player1.reset(game)
    player2.reset(game)

    while True:
        player1.contemplate(thinking_time)
        a = player1.choose()
        game.make_move(ind2move(a))
        
        # qtttgym.display.displayBoard(game)
        player1.sync(a)
        player2.sync(a)
        assert player1.root == player2.root
        # Check terminal
        winner = check_win(game)
        if winner is not None or len(game.moves) == 9: break 
        
        player2.contemplate(thinking_time)
        a = player2.choose()
        game.make_move(ind2move(a))
        winner = check_win(game)
        if winner is not None or len(game.moves) == 9: break 
        
        # qtttgym.display.displayBoard(game)
        player1.sync(a)
        player2.sync(a)
        assert player1.root == player2.root
    
    return winner

if __name__ == "__main__":
    strat1 = MCTS(rollouts=30)
    strat2 = MCTS(rollouts=3000)
    
    n_draws = 0
    strat1_wins = 0
    strat2_wins = 0
    for i in range(100):
        winner = play_game(strat1, strat2, thinking_time=math.inf)
        if winner:
            strat1_wins += 1
        elif winner == False:
            strat2_wins += 1
        elif winner == None:
            n_draws += 1
        num_games = 2*i + 1
        strat1_winrate = strat1_wins/num_games
        strat2_winrate = strat2_wins/num_games
        draw_rate = n_draws/num_games
        print(f"{strat1_winrate:.2f}, {strat2_winrate:.2f}, {draw_rate:.2f}, {num_games}")
        
        winner = play_game(strat2, strat1, thinking_time=math.inf)
        if winner:
            strat2_wins += 1
        elif winner == False:
            strat1_wins += 1
        elif winner == None:
            n_draws += 1
        num_games = 2*i + 2
        strat1_winrate = strat1_wins/num_games
        strat2_winrate = strat2_wins/num_games
        draw_rate = n_draws/num_games
        print(f"{strat1_winrate:.2f}, {strat2_winrate:.2f}, {draw_rate:.2f}, {num_games}")