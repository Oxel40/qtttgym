import math
from copy import deepcopy
import time
import pickle
import datetime

import numpy as np
import torch as pt
from torch.distributions import Categorical
from tqdm import trange
from torch.utils.tensorboard.writer import SummaryWriter

import qtttgym
from strategy import Strategy
from nn import QNet, VNet, Model

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

class CircularBuffer():
    def __init__(self, buf_size:int) -> None:
        self.bufsize = buf_size
        self.k = 0
        self.n = 0

        self.state_list = np.zeros(buf_size, dtype=object)
        self.hashes = set()
    
    def add(self, s):
        if s in self.hashes:
            return
        k = self.k
        if self.state_list[k] != 0:
            self.hashes.remove(self.state_list[k])
        self.state_list[k] = s
        self.hashes.add(s)
        
        self.k = (k + 1) % self.bufsize
        self.n = min(self.n + 1, self.bufsize)
        
    
    def get_batch(self, batchsize:int):
        i = np.random.randint(0, self.n, batchsize)
        return self.state_list[i]

    def load(self):
        try:
            self.hashes = pickle.load(open("states.pkl", 'rb'))
            self.n = min(len(self.hashes), self.bufsize)
            self.k = self.n % self.bufsize
            self.state_list[:self.n] = np.array(list(self.hashes))
            print(f"Sucessfully loaded {self.n} states")
        except:
            self.n = 0
            self.hashes = set()
            print("Loading buffer failed")

    def save(self):
        pickle.dump(self.hashes, open("states.pkl", "wb"))
    
    def __len__(self):
        return self.n

class MiniMax(Strategy):
    
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
            
            # MiniMax Property
            self.Q:dict = {a:0. for a in self.actions}
            self.explored = False 

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
            self.Q:dict = {a : 0 for a in self.actions}
        
        def get_actions(self):
            actions = list()
            for i in range(36):
                move = ind2move(i)
                if self.board[move[0]] != -1 or self.board[move[1]] != -1:
                    continue
                actions.append(i)
            return actions

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
            classic_state = np.zeros((9, 10), dtype=float)
            for i in range(9):
                classic_state[i][self.board[i]] = 1
            quantum_state = np.zeros((9, 9), dtype=float)
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
            turn = len(self.moves) == 0
            v = np.concatenate(
                (classic_state.reshape(-1), 
                 quantum_state.reshape(-1),
                 np.array([turn])))
            return v
        
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
            return f"GameState({self.board}, {self.winner})"
    

    def __init__(self, train=False, load_nn=True) -> None:
        super().__init__()
        self.Q = QNet(1, lr=1e-3)
        self.model = Model(1, lr=1e-4)
        self.model_slow = Model(1)
        self.advesary = Model(1)
        # self.Q:QNet = pt.compile(self.Q)
        # self.model:Model = pt.compile(self.model)
        # self.model_slow:Model = pt.compile(self.model_slow)
        self.alpha = 0.
        self.nodes:dict[int, self.GameState] = dict()
        self.tau = 0.005
        self.tau2 = 0.#01*self.tau
        self.buffer = CircularBuffer(1000000)
        self.train_mode = train
        self.batch_size = 64
        self.advesary_sync_time = 200000
        if train:
            self.writer = SummaryWriter()
        self.it = 0
        if load_nn:
            self.load()

    @pt.no_grad()
    def choose(self, s:GameState=None):
        if s is None:
            s = self.GameState.to_vector(self.game)
        else:
            s = s.to_vector()
        _, logits = self.model.forward(s)
        cat = Categorical(logits=logits)
        a = cat.sample()
        return int(a)
    
    @pt.no_grad()
    def adversary_choose(self, s:GameState):
        s = s.to_vector()
        _, logits = self.advesary.forward(s)
        cat = Categorical(logits=logits)
        return int(cat.sample())
    
    def reset(self, game:qtttgym.Board):
        self.game = game

    def train(self, n_rollouts:int=100, sgd_iters:int=1000, rollout_loadbar=False, sgd_loadbar=False):
        if rollout_loadbar:
            lbar = trange(n_rollouts, ncols=100)
        else:
            lbar = range(n_rollouts)
        v = 0
        for r in lbar:
            v += self.rollout()
        if self.train_mode:
            self.writer.add_scalar("value/WinLoss", v/n_rollouts, self.it)
        self.train_agent(sgd_iters, sgd_loadbar)
    
    def calc_td_target(self, s_batch_raw:list[GameState]):
        target = pt.zeros(len(s_batch_raw))
        states = pt.zeros((len(s_batch_raw), 172))
        actions = pt.zeros(len(s_batch_raw), dtype=pt.long)
        mask = pt.ones(len(s_batch_raw), dtype=bool)
        for i, s in enumerate(s_batch_raw):
            if s.terminal: 
                target[i] = 0
                mask[i] = False
                continue
            a = np.random.choice(s.actions)
            states[i] = pt.tensor(s.to_vector())
            actions[i] = a
            sn = self._step(s, a)
            if sn.terminal:
                r = self._reward(sn)
                r = r if s.turn else -r
                target[i] += r
                continue
            an = self.adversary_choose(sn)
            snn = self._step(sn, an)
            if snn.terminal:
                r = self._reward(snn)
                r = r if s.turn else -r
                target[i] += r
            else:
                vnn, _ = self.model_slow.forward(snn.to_vector())
                target[i] += vnn.detach().squeeze(-1)
        return states[mask], actions[mask], target[mask]

    @pt.no_grad()
    def calc_oracle_td(self, s_batch_raw:list[GameState]):
        target = pt.zeros(len(s_batch_raw))
        states = pt.zeros((len(s_batch_raw), 172))
        actions = pt.zeros(len(s_batch_raw), dtype=pt.long)
        mask = pt.ones(len(s_batch_raw), dtype=bool)
        for i, s in enumerate(s_batch_raw):
            if s.terminal: 
                target[i] = 0
                mask[i] = False
                continue
            states[i] = pt.tensor(s.to_vector())
            a = np.random.choice(s.actions)
            actions[i] = a
            # _, logits = self.model.forward(s.to_vector())
            # cat = Categorical(logits=logits)
            # for a in s.actions:
            sn = self._step(s, a)
            if sn.terminal:
                r = self._reward(sn)
                r = r if s.turn else -r
                target[i] += r
                continue
            _, logitsn = self.advesary.forward(sn.to_vector())
            catn = Categorical(logits=logitsn)
            vnn = 0
            for an in sn.actions:
                snn = self._step(sn, an)
                if snn.terminal:
                    r = self._reward(snn)
                    r = r if s.turn else -r
                    vnn += r * catn.probs[an]
                else:
                    vnn_, _ = self.model_slow.forward(snn.to_vector())
                    vnn += vnn_.detach().squeeze(-1) * catn.probs[an]
            target[i] = vnn
        return states[mask], actions[mask], target[mask]

    def calc_qloss(self, td_target:pt.Tensor, s, a):
        q = self.Q.forward(s, a).squeeze(-1)
        adv = td_target - q
        # print(td_target)
        # print(q)
        # input()
        return 0.5 * adv.pow(2)
    
    def calc_vloss(self, s) -> pt.Tensor:
        v, logits = self.model.forward(s)
        cat = Categorical(logits=logits)
        a = cat.sample()
        with pt.no_grad():
            q = self.Q.forward(s, a)
            ent:pt.Tensor = cat.entropy()
        v_target = q + self.alpha * ent
        return 0.5 * (v - v_target).pow(2)
    
    def calc_fullpiloss(self, s) -> pt.Tensor:
        v, logits = self.model.forward(s)
        v  = v.detach()
        mask = self.model.get_mask(s)
        all_actions = np.arange(36)
        # with pt.no_grad():
        q = self.Q.forward(s[:, None], all_actions).squeeze(-1).detach()
        q[mask] -= pt.inf
        logp = logits - pt.logsumexp(logits, dim=-1, keepdim=True)
        min_real = pt.finfo(logp.dtype).min
        logp, q = pt.clamp(logp, min_real), pt.clamp(q, min_real)
        if self.alpha == 0.:
            a_best = pt.argmax(q, dim=-1, keepdim=True)
            DKL = -pt.gather(logp, -1, a_best).squeeze(-1)
            return DKL
        logZ = pt.logsumexp((q-v)/self.alpha, dim=-1)

        CE = pt.sum(logp.exp() * (q-v), dim=-1)
        ent = pt.sum(-logp.exp() * logp, dim=-1)
        DKL = self.alpha * (ent + logZ) - CE
        return DKL

    def calc_surrpiloss(self, s) -> pt.Tensor:
        v, logits = self.model.forward(s)
        cat = Categorical(logits=logits)
        a = cat.sample()
        logp = cat.log_prob(a)
        ent = cat.entropy()
        with pt.no_grad():
            q = self.Q.forward(s, a).squeeze(-1)
            Adv = q + self.alpha * ent - v.squeeze(-1)
        piloss = -Adv * logp - self.alpha * ent
        return piloss
        
    
    def train_agent(self, num_iters, sgd_loadbar=True):
        all_actions = np.arange(36)
        game = qtttgym.Board(qtttgym.QEvalClassic())
        root = self.GameState(game.board, game.moves, True, None, False).to_vector()

        data = {
            "loss":[],
            "U":[],
            "U_":[]
        }
        if self.train_mode:
            self.writer.add_scalar("stats/num_states", len(self.buffer), self.it)
        if sgd_loadbar:
            EPOCHS = trange(num_iters, ncols=150)
        else:
            EPOCHS = range(num_iters)
        for _ in EPOCHS:
            s_batch:list[self.GameState] = self.buffer.get_batch(self.batch_size)
            # s_batch, a_batch, q_target = self.calc_td_target(s_batch)
            s_batch, a_batch, q_target = self.calc_oracle_td(s_batch)
            # print(q_target)
            # quit()
            qloss = self.calc_qloss(q_target, s_batch, a_batch)

            qloss = qloss.mean()
            self.Q.optim.zero_grad()
            qloss.backward()
            self.Q.optim.step()

            vloss = self.calc_vloss(s_batch)
            # piloss = self.calc_surrpiloss(s_batch)
            piloss = self.calc_fullpiloss(s_batch)

            vloss = vloss.mean()
            piloss = piloss.mean()
            self.model.optim.zero_grad()
            (10*vloss + piloss).backward()
            self.model.optim.step()
            
            self.slow_update()
            q = self.Q.forward(root[None, :], all_actions)
            v, _ = self.model(root)
            topa = sorted(list(all_actions), key=lambda k: q[k, 0], reverse=True)
            q = q[topa]
            if sgd_loadbar:
                EPOCHS.set_description_str(f"{float(qloss):.2e}, {float(piloss):.2e} | {float(v[0]):.2f}| {ind2move(topa[0])}: {q[0,0]:.2f}, {ind2move(topa[-1])}: {q[-1,0]:.2f}")

            if self.train_mode:
                self.writer.add_scalar("loss/qloss", float(qloss), self.it)
                self.writer.add_scalar("loss/vloss", float(vloss), self.it)
                self.writer.add_scalar("loss/piloss", float(piloss), self.it)
                self.writer.add_scalar("value/V", float(v), self.it)
                self.writer.add_scalar("value/best 1st move", float(q[0]), self.it)
                self.writer.add_scalar("value/worst 1st move", float(q[-1]), self.it)
                self.it += 1
                if (self.it + 1) % self.advesary_sync_time == 0:
                    self.advesary.load_state_dict(self.model.state_dict())
                    self.model.optim = pt.optim.Adam(self.model.parameters(), lr=1e-4)
                    self.Q.optim = pt.optim.Adam(self.model.parameters(), lr=1e-3)
                    
            data["loss"].append(float(qloss))
            data["U"].append(float(q[0, 0]))
            data["U_"].append(float(q[-1, 0]))
        return data

    @pt.no_grad()
    def slow_update(self):
        for source, target in zip(self.model.parameters(), self.model_slow.parameters()):
            target.data = target * (1 - self.tau) + source * self.tau
        for source, target in zip(self.model.parameters(), self.advesary.parameters()):
            target.data = target * (1 - self.tau2) + source * self.tau2
        

    def get_mask(self, s:pt.Tensor):
        classic_state = s[..., :90].reshape((s.shape[0], 9, 10))
        occupied = classic_state[..., :-1].any(dim=-1)
        mask = pt.zeros(s.shape[:-1] + (36,), dtype=bool)
        for a in range(36):
            i, j = ind2move(a)
            mask[..., a] = pt.logical_or(occupied[..., i], occupied[..., j])
        return mask
    
    @staticmethod
    def _reward(node:GameState):
        r = 0
        if node.winner is None:
            return 0
        if node.winner:
            r = 1
        else:
            r = -1
        return r
        
    def _step(self, node:GameState, action) -> GameState:
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
        new_node.update_actions()
        return new_node
    
    @pt.no_grad()
    def rollout(self):
        game = qtttgym.Board(qtttgym.QEvalClassic())
        node = self.GameState(game.board, game.moves, True, None, False)
        self.buffer.add(node)
        isplayer1 = np.random.uniform(0, 1) < 0.5
        while not node.terminal:
            if not (isplayer1 ^ node.turn):
                a = self.choose(node)
            else:
                a = self.adversary_choose(node)

            sn = self._step(node, a)
            self.buffer.add(sn)
            node = sn
        r = self._reward(node)
        r = r if isplayer1 else -r
        # if self.train_mode:
        #     self.writer.add_scalar("value/r", r, self.it)
        return r
        
            
    def save(self):
        pt.save(self.Q.state_dict(), "qnet.pt")
        pt.save(self.model.state_dict(), "net.pt")
        self.buffer.save()

    def load(self, load_buffer=True):
        try:
            self.Q.load_state_dict(pt.load("qnet.pt"))
            self.model.load_state_dict(pt.load("net.pt"))
            self.model_slow.load_state_dict(self.model.state_dict())
            self.advesary.load_state_dict(self.model.state_dict())
            print("models successfully loaded")
        except:
            pass
        if load_buffer:
            self.buffer.load()
        
    def __str__(self) -> str:
        return "MiniMax()"

if __name__ == "__main__":
    strat = MiniMax(train=True, load_nn=True)
    # strat.load()
    for _ in range(1000000):
        strat.train(1000, 100, sgd_loadbar=False)
        strat.save()
    quit()
    data = strat.train_agent(10000)
    strat.save()

    import matplotlib.pyplot as plt
    
    plt.plot(data["U"], label="U")
    plt.plot(data["U_"], label="U_")
    plt.grid(True)
    plt.legend()
    plt.savefig("Winner")
    plt.show()
    