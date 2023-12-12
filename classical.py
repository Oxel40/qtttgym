import numpy as np
import math
from tqdm import trange

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

            # Tree search reference counter
            self.ref_count = 0

        def __hash__(self) -> int:
            return hash(self.board) # + (self.turn, self.winner, self.terminal))
        
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
        self.root.ref_count += 1

        self.c_puct = 1
        self.nodes:dict[int, self.GameState] = dict()
        self.nodes[hash(self.root)] = self.root

    def make_move(self, action):
        if action not in self.root.children:
            raise Exception("Invalid Action")
        if self.root.children[action] is None:
            # node has not been fully expanded
            # only expand the child of this action
            print("Expanding in MOVE")
            self._expand_child(self.root, action)
        
        new_root = self.root.children[action]
        for child in self.root.children.values():
            if hash(child) == hash(new_root): continue
            self.prune_borrowcheck(child)

        self.root.ref_count -= 1
        if self.root.ref_count == 0:
            self.nodes.pop(hash(self.root))
        self.root = new_root

    def prune_borrowcheck(self, node:GameState):
        # subtract 1 from the borrow checkers
        # if a borrow checker hits 0
        # remove the node from memory
        if node is None:
            return
        
        for child in node.children.values():
            self.prune_borrowcheck(child)
        if node.ref_count == 0:
            print(node)
            input()
        node.ref_count -= 1
        if node.ref_count == 0:
            self.nodes.pop(hash(node))
            # del self.nodes[hash(node)]

    def choose(self):
        n = self.root
        def score(a):
            if n.N[a] == 0:
                return -math.inf
            return n.Q[a]

        a_best = max(self.root.actions, key=score)
        return a_best
    
    def _expand_child(self, node:GameState, action):
        new_node = self._step(node, action)
        # print(new_node)
        node_hash = hash(new_node)
        if node_hash in self.nodes:
            # print("CACHE HIT")
            # print(new_node)
            # print(self.nodes[node_hash])
            # print(new_node.N_tot, self.nodes[node_hash].N_tot)
            # input()
            new_node = self.nodes[node_hash]

        else:
            self.nodes[node_hash] = new_node
        #     print("CACHE MISS")
        # input()
        node.children[action] = new_node
        new_node.ref_count += 1
        

    def _step(self, node:GameState, action) -> GameState:
        new_board = node.board[:action] + (node.turn,) + node.board[action + 1 :]
        turn = not node.turn
        winner = _find_winner(new_board)
        is_terminal = (winner is not None) or not any(v is None for v in new_board)
        new_state = self.GameState(new_board, turn, winner, is_terminal)
        return new_state

    def do_rollout(self):
        "Make the tree one layer better. (Train for one iteration.)"
        # select until we uct select a new node
        path, leaf = self._select(self.root)
        reward = self._simulate(leaf)
        path.append((leaf, None))
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

    def _simulate(self, node:GameState) -> float:
        # Simulates from the node to the bottom
        invert_reward = True
        while not node.terminal:
            if node.P is None:
                # Evaluate the action probabilities of this Node
                node.P = self.get_action_probs(node)
            # Sample an action according to P
            a = self.sample_action(node)
            node = self._step(node, a)
            # node = node.children[a]
            # node = self._step(node, a)
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
            if node.N[a] == 0:
                return math.inf
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

def check_refcount(node:TTTGame.GameState):
    if node is None:
        return
    print(node)
    print(node.ref_count)
    for child in node.children.values():
        check_refcount(child)

if __name__ == "__main__":
    # play_game()
    game = TTTGame()
    # game.root = game.GameState((None, True, None, False, False, True, True, None, False), True, None, False)
    print(game.root)
    n = 0
    while not game.root.terminal:
        # a = int(input("Make move: "))
        # game.make_move(a)
        # print(game.root)
        rollout_bar = trange(5000, ncols=150)
        for i in rollout_bar:
            game.do_rollout()
            node = game.root
            c_expl = [node.N[a] for a in node.actions]
            c_expl = " ".join([f"{x}" for x in c_expl])
            q = " ".join([f"{x:.3f}" for x in node.Q.values()])
            rollout_bar.set_description_str(f"{q} | {c_expl} | {len(game.nodes)}")
        # print(game.nodes)
            # input()
        # print(len(game.nodes))
        # check_refcount(game.root)
        a = game.choose()
        game.make_move(a)
        print(game.root)
        input()
        # # quit()
        n += 1