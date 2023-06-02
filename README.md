# qtttgym

AI Gymnasium like environment for QTTT.

Derived from the quantum tic-tac-toe game described in [this paper](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-320190).

## Theory

We want to optimize for two policies $\pi_1$ and $\pi_2$. The state space contains the current board state which contains the ghost pieces for both players, the classical game board and whose turn it is.

$$
\mathcal{S} = \mathbb{N}_9 ^2 \times \mathbb{N}_9 ^2 \times \mathbb{N}_9^9 \times \mathbb{N}_2
$$

The reward is given at the end of the game, player 1 is awarded a reward of 1 if they win, player 2 is awarded a reward of -1 if they win.

Where $\mathbb{N}_9$ is notation for the set of 9 integers 1,2... 9. This is a finite horizon MDP with a terminal state so we will define the value function as the sum of rewards(no discount factor). We can then define this zero sum game as the following

$$
\max_{\pi_1}\min_{\pi_2} V^{\pi_1, \pi_2}(s) = \mathbb{E}\big[\sum_t r(s_t, a_t) \big| s_0 = s\big]
$$

This minimax one policy is attempting to maximize the value and the other is attempting to minimize the value. With this we should effectivly use all of the traditional RL framework but we simply add a negation for player 2.

## Action space and NN architecture

So an agent has a total of $9^2=81$ different actions. This on the limit of what is doable with traditional DQN. We might need to use an policy gradient approach to this problem where use make create an autoregressive policy. Since our policy has to generate two integers $x$ and $y$ we can utilize the factorization of distributions.

$$
\pi_\theta(x,y|s) = \pi_\theta(x|y,s)\pi_\theta(y|s)
$$

This way we only need to design a NN with 18 output nodes, which is significantly easier to train.

![image](mdp1.png)

Since our environment is relatively small and easy to sample from, we should be able to simply run PPO on this problem and hopefully it should converge to a good policy. The potential downside I see with this is that if we train an agent with self play, we might run into situation where our AI might not be able to know how to play against any other opponent that itself. So if we were to do a Q learning approach(like SAC) we might be more flexible in situations where the AI is not playing against itself. In this case I think we can be really sneaky and implement a really clever Q function NN that might be easier to train.

![image](q_nn.png)

In short, the action encoding will since we shouldn't care about the action of player 2 on player 1s turn, we should completly remove the latent encoding of that action as we propogate our latent vector foward through our NN.

These are all of my current thoughts on this problem
