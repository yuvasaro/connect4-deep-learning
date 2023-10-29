"""Connect 4 board tester."""
# Run command: python -m unittest test.test_board

import unittest

import numpy as np

from c4.board import Board
from c4.constants import P1, P2


class TestBoard(unittest.TestCase):
    
    def setUp(self):
        self.board = Board()

    def test_valid_moves(self):
        self.board.place_coin(P1, (0, 1))
        self.board.place_coin(P1, (0, 3))
        self.board.place_coin(P1, (1, 3))
        self.board.place_coin(P1, (0, 4))
        
        self.assertEqual(self.board.valid_moves(), [(0, 0), (0, 2), (0, 5), (0, 6), (1, 1), (1, 4), (2, 3)])

    def test_place_coin(self):
        self.board.place_coin(P1, (0, 0))
        self.board.place_coin(P1, (0, 1))
        self.board.place_coin(P1, (0, 2))
        self.board.place_coin(P1, (0, 3))

        np.testing.assert_equal(self.board._board, np.array([[1, 1, 1, 1, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 0]]))
        
    def test_check_win(self):
        self.board.place_coin(P2, (0, 0))
        self.board.place_coin(P2, (1, 1))
        self.board.place_coin(P2, (2, 2))
        self.board.place_coin(P2, (3, 3))

        self.assertEqual(self.board.check_win(P2), True)


if __name__ == "__main__":
    unittest.main()
