from abc import abstractclassmethod

class Strategy:
    def __init__(self) -> None:
        ...

    def reset(self, game):
        """reset the game to the starting state
        """
        self.game = game
    
    @abstractclassmethod
    def contemplate(self, thinking_time:int):
        """spend some time planning your move

        Args:
            time (int): time limit 
        """

    @abstractclassmethod
    def choose(self):
        """make a choice
        Return:
            a (int): action
        """
        ...
    
    @abstractclassmethod
    def sync(self, a):
        """Sync your strategy given the move that was just made

        Args:
            a (int): action that was taken by your opponent
            sn (_type_): next state that was transitioned too
        """
        ...
