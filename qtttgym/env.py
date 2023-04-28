from qtttgym import Board
from qtttgym import QEvalClassic
from qtttgym import displayBoard


class ActType:
    ...


class ObsType:
    ...


class RewType:
    ...


class Env():
    def __init__(self):
        self.action_space = [(a, b) for a in range(10) for b in range(10)]
        self.observation_space = list(range(10)).extend(
            [(a, b) for a in range(12) for b in range(10)]
        )  # TODO
        self._p2_turn = False
        self._turn = 0
        self._gameboard = Board(QEvalClassic())

    def step(self, action: ActType, verbose=False) -> tuple[ObsType, RewType, bool, bool]:
        try:
            squareFirst = action[0]
            squareSecond = action[1]

            self.board.make_move((squareFirst, squareSecond))

            self._p2_turn = not self._p2_turn
            self._turn += 1  # TODO fix last move thingy
        except Exception as e:
            if verbose:
                print('noop (i.e. invalid) move...', e)
        obs = self._observation()
        rew = self._reward()
        terminated = any(map(lambda r: r != 0, rew)) or self._turn > 8
        truncated = False
        return (obs, rew, terminated, truncated)

    def reset(self) -> ObsType:
        self.__init__()
        return self._observation()

    def render(self):
        displayBoard(self.board)

    def observ(self):
        return self._observation()

    def _observation(self):
        out = self._gameboard.moves
        for i, r in enumerate(self._gameboard.board):
            if r >= 0:
                out[r] = i
        return tuple(out)

    def _reward(self):
        """
        1 for a win
        -1 for a loss
        0 for a draw or if game is not done
        """
        p1_round, p2_round = self._gameboard.check_win()
        p1_rew = 0
        p2_rew = 0
        if p1_round >= 0 or p2_round >= 0:
            # Make sure that a -1 from check_win doesn't count as a win
            if p1_round < 0:
                p1_round = 10
            if p2_round < 0:
                p2_round = 10

            p1_rew = 2*int(p1_round < p2_round) - 1
            p2_rew = 2*int(p1_round > p2_round) - 2

        return (p1_rew, p2_rew)
