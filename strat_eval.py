import qtttgym
import math

from mcts import MCTS
from alphazero import AlphaZero
from strategy import Strategy

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

def eval_strats(strat1:Strategy, strat2:Strategy, num_games:int=200, thinking_time:int=30):
    n_draws = 0
    strat1_wins = 0
    strat2_wins = 0
    for i in range(num_games//2):
        winner = play_game(strat1, strat2, thinking_time=thinking_time)
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
        print(f"{strat1_winrate:.2f}, {strat2_winrate:.2f}, {draw_rate:.2f}, {num_games}", end="\r")
        
        winner = play_game(strat2, strat1, thinking_time=thinking_time)
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
        print(f"{strat1_winrate:.2f}, {strat2_winrate:.2f}, {draw_rate:.2f}, {num_games}", end="\r")
    print()


if __name__ == "__main__":
    print("Evaluating: AlphaZero(300) vs MCTS(3000)")
    eval_strats(AlphaZero(rollouts=300), 
                MCTS(rollouts=3000), 
                thinking_time=math.inf)