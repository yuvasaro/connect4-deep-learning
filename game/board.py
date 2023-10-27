"""A Connect 4 Board class."""

import numpy as np


class Board:
    """The Connect 4 game board."""

    def __init__(self):
        self.board = np.zeros((6, 7))
