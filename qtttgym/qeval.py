import random


class QEvalClassic:
    def eval(self, entangled_moves):
        out = [-1] * len(entangled_moves)
        # O(n)
        move_inx = dict()
        for i, move in enumerate(entangled_moves):
            move_inx[move] = i

        rutor = []
        # O(1)
        for i in range(9):
            rutor.append(set())
        # O(n)
        for move in entangled_moves:
            rutor[move[0]].add(move)
            rutor[move[1]].add(move)

        # O(n)
        # Evaluate moves not directly in the entangled cycle
        for i in range(9):
            while len(rutor[i]) == 1:
                move = rutor[i].pop()
                index = 1 if i == move[0] else 0
                next_i = move[index]
                move_res = move[1-index]
                out[move_inx[move]] = move_res  # O(1)
                rutor[next_i].remove(move)  # O(1)
                i = next_i

        # O(n)
        # Collapse cyclicly dependent quantum moves
        out[-1] = random.choice(entangled_moves[-1][0:2])
        last_move = entangled_moves[-1]
        r_start = last_move[0]
        r = last_move[1]
        fell_in_r = r == out[-1]
        rutor[r].remove(last_move)  # O(1)
        while r != r_start:
            move = rutor[r].pop()  # O(1)

            move_res = move[0] if fell_in_r ^ (move[0] == r) else move[1]
            out[move_inx[move]] = move_res  # O(1)

            r = move[0] if move[1] == r else move[1]
            rutor[r].remove(move)  # O(1)
            fell_in_r = r == move_res

        return out
