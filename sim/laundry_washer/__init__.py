"""Front-loading laundry environment.

No trained policy is loaded. Garments are rigid proxies. Detergent is a
pose-and-dose check, not a fluid. The CPU classes are the task layer.
Newton / Isaac execution is optional and is not imported here.
"""

from .env import LaundryEnv

__all__ = ["LaundryEnv"]
