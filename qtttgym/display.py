from qtttgym import Board


def displayBoard(board: Board):
    list_of_buffers = [[' '] * 9 for _ in range(9)]

    for i, m in enumerate(board.moves):
        list_of_buffers[m[0]][i] = str(i)
        list_of_buffers[m[1]][i] = str(i)

    for i, b in enumerate(board.board):
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
    print(out_string)
