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
from torch.distributions import Categorical
from collections import namedtuple
from random import choice
import numpy as np
import math
from tqdm import trange
import qtttgym
import random
from nn import Model
from copy import deepcopy
import torch as pt
from qttt import QTTTGame

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

def play_game(net:Model, n_rollouts=10):
    game = QTTTGame()
    def action_probs(node:QTTTGame.GameState):
        _, logits = net.forward(node.to_vector())
        cat = Categorical(logits=logits)
        p = cat.probs
        node.dist = cat
        return {a: float(p[a]) for a in node.actions}

    def get_action(node:QTTTGame.GameState):
        return int(node.dist.sample())
    
    game.get_action_probs = action_probs
    game.sample_action = get_action
    
    # game.get_action_probs
    nodes = [game.root]
    while not game.root.terminal:
        rollout_bar = range(n_rollouts)
        for i in rollout_bar:
            game.do_rollout()
            # node = game.root
            # qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)
            # q = " ".join([f"{len(game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
            # rollout_bar.set_description_str(f"{q}")
        a = game.choose()
        game.make_move(a)
        nodes.append(game.root)
    # for s in nodes:
    #     print(s)
    # quit()
    # print(game.root)
        # input()
    return nodes, game.root.winner

def expand_symetries(states, actions):
    # Return all symmetric duplicates
    ...

    

def play_vs_ai(net:Model):
    # ai_game = QTTTGame()
    # def action_probs(node:QTTTGame.GameState):
    #     _, logits = net.forward(node.to_vector())
    #     cat = Categorical(logits=logits)
    #     p = cat.probs
    #     node.dist = cat
    #     return {a: float(p[a]) for a in node.actions}
    # def get_action(node:QTTTGame.GameState):
    #     return int(node.dist.sample())
    # ai_game.get_action_probs = action_probs
    # ai_game.sample_action = get_action

    game = QTTTGame()

    print(game.root)
    while not game.root.terminal:
        node = game.root
        # rollout_bar = trange(300)
        # for i in rollout_bar:
        #     game.do_rollout()
        #     qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)
        #     q = " ".join([f"{len(ai_game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
        #     rollout_bar.set_description_str(f"{q}")
        # a = ai_game.choose()
        _, logits = net.forward(game.root.to_vector())
        cat = Categorical(logits=logits)
        a = int(cat.sample())
        game.make_move(a)
        print(game.root)

        if game.root.terminal: break
        rollout_bar = trange(3000)
        for i in rollout_bar:
            game.do_rollout()
            node = game.root
            qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)
            q = " ".join([f"{len(game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
            rollout_bar.set_description_str(f"{q}")
        a = game.choose()
        game.make_move(a)

        print(game.root)

    return game.root.winner
    
def play_vs_ai2(net:Model):
    # ai_game = QTTTGame()
    # def action_probs(node:QTTTGame.GameState):
    #     _, logits = net.forward(node.to_vector())
    #     cat = Categorical(logits=logits)
    #     p = cat.probs
    #     node.dist = cat
    #     return {a: float(p[a]) for a in node.actions}
    # def get_action(node:QTTTGame.GameState):
    #     return int(node.dist.sample())
    # ai_game.get_action_probs = action_probs
    # ai_game.sample_action = get_action

    game = QTTTGame()

    print(game.root)
    while not game.root.terminal:
        node = game.root
        # rollout_bar = trange(300)
        # for i in rollout_bar:
        #     game.do_rollout()
        #     qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)
        #     q = " ".join([f"{len(ai_game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
        #     rollout_bar.set_description_str(f"{q}")
        # a = ai_game.choose()
        rollout_bar = trange(3000)
        for i in rollout_bar:
            game.do_rollout()
            node = game.root
            qvals = sorted(list(node.Q.items()), key=lambda x: x[1], reverse=True)
            q = " ".join([f"{len(game.root.moves)} | {ind2move(x[0])}:{x[1]:.3f}" for x in qvals[:5]])
            rollout_bar.set_description_str(f"{q}")
        a = game.choose()
        game.make_move(a)
        print(game.root)

        if game.root.terminal: break
        _, logits = net.forward(game.root.to_vector())
        cat = Categorical(logits=logits)
        a = int(cat.sample())
        game.make_move(a)

        print(game.root)

    return game.root.winner

if __name__ == "__main__":
    # play_game()
    # np.random.seed(1)
    # random.seed(1)
    net = Model()
    # net.load_state_dict(pt.load("model.pt"))
    # ai_wins = 0
    # ai_losses = 0
    # draws = 0
    # for _ in range(100):
    #     winner = play_vs_ai(net)
    #     # print(winner)
    #     if winner == True:
    #         ai_wins += 1
    #     elif winner == False:
    #         ai_losses += 1
    #     else:
    #         draws += 1
    #     winner = play_vs_ai2(net)
    #     # print(winner)
    #     if winner == True:
    #         ai_losses += 1
    #     elif winner == False:
    #         ai_wins += 1
    #     else:
    #         draws += 1

    #     print(ai_wins, ai_losses, draws)
    # quit()
    # quit()
    alpha = 1
    runs = 30
    decay = (0.01/alpha) **(1/runs)
    for run in range(runs):
        M = 50
        s_batch = []
        pi_batch = []
        mask_batch = []
        v_batch = []
        done = []
        n_rollouts = 100
        for _ in trange(M, desc="Rollouts"):
            states, winner = play_game(net, n_rollouts)
            v_target = 0
            if winner:
                v_target = 1
            elif winner:
                v_target = -1
            for i, n in enumerate(states):

                s_batch.append(n.to_vector())
                if i == len(states) - 1:
                    pi_batch.append(np.ones(36)/36)
                    mask_batch.append(np.ones(36, dtype=bool))
                    done.append(True)
                else:
                    a = np.array(n.actions, dtype=int)
                    pi = np.zeros(36)
                    pi[a] = (np.array(list(n.N.values()))/n_rollouts)**alpha
                    pi /= np.sum(pi, axis=-1)
                    pi_batch.append(pi)
                    mask_batch.append(n.action_mask())
                    done.append(False)
                v_batch.append(v_target)
                v_target = -v_target

        s_batch = pt.tensor(np.array(s_batch))
        pi_batch = pt.tensor(np.array(pi_batch))
        v_batch = pt.tensor(v_batch)
        mask_batch = pt.tensor(np.array(mask_batch))
        not_done = pt.tensor(done, dtype=bool).logical_not()
        # print(s_batch.shape, a_batch.shape, v_batch.shape, not_done.shape)
        EPOCHS = trange(50)
        for ep in EPOCHS:
            v, logits = net.forward(s_batch)
            logits = logits[not_done]
            pi = pi_batch[not_done]
            logp = pt.log_softmax(logits, dim=-1)
            mask = mask_batch[not_done]
            l = pi[mask] * logp[mask]
            J = pt.zeros_like(pi)
            J[mask] = pi[mask] * (pt.log(pi[mask] + 1e-7) - logp[mask])
            J = J.sum(-1)
            L = 0.5*(v - v_batch).pow(2)
            loss = L.mean() + J.mean(0)
            net.optim.zero_grad()
            loss.backward()
            net.optim.step()
            EPOCHS.set_description_str(f"L: {float(L.mean()):.2f}, J: {float(J.mean()):.2f}")
        # alpha *= decay
        pt.save(net.state_dict(), "model.pt")