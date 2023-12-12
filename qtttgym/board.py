class Board:
    def __init__(self, qevaluator):
        # TODO: might be able to use int instead of str
        self.moves: list[tuple[int, int, str]] = []
        self.board: list[int] = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.qstructs: list[set] = []
        self.qeval = qevaluator

    def make_move(self, move: tuple[int, int]):
        if (move[0] == move[1]):
            raise Exception(
                "Move in same square not allowed when not necessary")

        if self.board[move[0]] != -1 or self.board[move[1]] != -1:
            raise Exception("Move in classical square not allowed")
        if move[0] > move[1]:
            # This fix makes sure states are hashable
            move = (move[1], move[0])
        self.moves.append((move[0], move[1], len(self.moves)))
        self.update_qstructs(move)
        # Autofill last square
        if self.board.count(-1) == 1:
            idx = self.board.index(-1)
            self.board[idx] = len(self.moves)
            self.moves.append((idx, idx, len(self.moves)))

    def update_qstructs(self, move: tuple):
        m0 = -1
        for i in range(len(self.qstructs)):
            current_set = self.qstructs[i]
            if move[0] in current_set:
                m0 = i
                break

        m1 = -2
        for j in range(len(self.qstructs)):
            current_set = self.qstructs[j]
            if move[1] in current_set:
                m1 = j
                break

        if m0 == m1:
            # Do quantum evaluation
            def filter_func(m):
                return m[1][0] in self.qstructs[m0]

            entangled_moves_zip = list(
                filter(filter_func, enumerate(self.moves)))
            entangled_moves = [j for _, j in entangled_moves_zip]
            rounds = [i for i, _ in entangled_moves_zip]
            outcomes = self.qeval.eval(entangled_moves)

            for r, o in zip(rounds, outcomes):
                self.board[o] = r

            self.qstructs.pop(m1)

        elif m0 >= 0 and m1 >= 0:
            # Unite sets
            self.qstructs[m0] = self.qstructs[m0].union(self.qstructs[m1])
            self.qstructs.pop(m1)
        else:
            # Add to sets
            i = max(m0, m1)
            if i < 0:
                self.qstructs.append(set())
                i = len(self.qstructs) - 1
            self.qstructs[i].add(move[0])
            self.qstructs[i].add(move[1])

    def check_win(self):
        """Return round of wincondition. -1 if no wincondition"""
        def map_func(m):
            if m < 0:
                return 0
            return (m % 2) * 2 - 1

        mark_list = list(map(map_func, self.board))

        # You can't win on round 10 (there is no round 10)
        p1_round = 10
        p2_round = 10

        # Check rows
        for i in range(3):
            s = sum(mark_list[i*3:(i+1)*3])
            if s == -3:
                p1_round = min(p1_round, max(self.board[i*3:(i+1)*3]))
            elif s == 3:
                p2_round = min(p2_round, max(self.board[i*3:(i+1)*3]))

        # Check cols
        for i in range(3):
            s = sum(mark_list[i::3])
            if s == -3:
                p1_round = min(p1_round, max(self.board[i::3]))
            elif s == 3:
                p2_round = min(p2_round, max(self.board[i::3]))

        # Check diagonal
        s = sum(mark_list[2:7:2])
        if s == -3:
            p1_round = min(p1_round, max(self.board[2:7:2]))
        elif s == 3:
            p2_round = min(p2_round, max(self.board[2:7:2]))
        s = sum(mark_list[0:9:4])
        if s == -3:
            p1_round = min(p1_round, max(self.board[0:9:4]))
        elif s == 3:
            p2_round = min(p2_round, max(self.board[0:9:4]))

        p1_round = p1_round if p1_round < 10 else -1
        p2_round = p2_round if p2_round < 10 else -1

        return p1_round, p2_round
