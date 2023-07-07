import qtttgym
import ray.rllib as rllib
import ray.rllib.utils

# board = qtttgym.Board(qtttgym.QEvalClassic())
env = qtttgym.Env()
ray.rllib.utils.check_env(qtttgym.Env())