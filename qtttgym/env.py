from qtttgym import Board
from qtttgym import QEvalClassic
from qtttgym import displayBoard


ActType = tuple[int, int]
ObsType = tuple[int | tuple[int, int, int], ...]
RewType = tuple[float, float]


class Env():
    def __init__(self):
        # TODO: fix action_space and observation_space
        self.action_space = [(a, b) for a in range(9) for b in range(a+1, 9)]
        self.observation_space = ...
        self._gameboard = Board(QEvalClassic())
        self._reward_map = {
            "win": 1.0,
            "loss": -1.0,
            "draw": 0.0,
            "otherwise": 0.0
        }
        
    def step(self, action: ActType, verbose=False) -> tuple[ObsType, RewType, bool, bool]:
        cur_player = self.turn % 2
        try:
            squareFirst = action[0]
            squareSecond = action[1]

            self._gameboard.make_move((squareFirst, squareSecond))
        except Exception as e:
            if verbose:
                print('noop (i.e. invalid) move...', e)
        # unsure about including the binary value for which players turn it is
        # this information is given implicitly in the state vector, but I'm not sure
        obs = self._observation() #+ (cur_player,)
        rew = self._reward()
        terminated = any(map(lambda r: r != 0, rew)) or self.turn() > 8
        truncated = False
        return (obs, rew, terminated, truncated)

    def reset(self) -> ObsType:
        self.__init__()
        return self._observation()

    def render(self):
        displayBoard(self._gameboard)

    def observ(self):
        return self._observation()

    def turn(self):
        return len(self._gameboard.moves)

    def _observation(self):
        q_states_player1 = []
        q_states_player2 = []
        classical_state = self._gameboard.board
        classical_pieces = set(classical_state)
        for move in self._gameboard.moves:
            if move[-1] not in classical_pieces:
                if move[-1] % 2:
                    q_states_player2.append(move[:-1])
                else:
                    q_states_player1.append(move[:-1])
        return (q_states_player1, q_states_player2, classical_state)

    def _reward(self):
        """
        1 for a win
        -1 for a loss
        0 for a draw or if game is not done
        """
        p1_round, p2_round = self._gameboard.check_win()
        p1_rew = self._reward_map["otherwise"]
        p2_rew = self._reward_map["otherwise"]

        # round is -1 if player is not in a win condition
        # handle -1, 10 is safe because we can't have 10 turns
        if p1_round < 0:
            p1_round = 10
        if p2_round < 0:
            p2_round = 10

        if p1_round < p2_round:
            p1_rew = self._reward_map["win"]
            p2_rew = self._reward_map["loss"]
        if p2_round < p1_round:
            p1_rew = self._reward_map["loss"]
            p2_rew = self._reward_map["win"]

        return (p1_rew, p2_rew)
