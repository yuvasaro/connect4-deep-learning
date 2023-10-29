"""Connect 4 game tester."""
# Rum command: python -m unittest test.test_game

import unittest

import numpy as np

from c4.constants import P1, P2
from c4.game import Game


class TestGame(unittest.TestCase):

    def setUp(self):
        self.game = Game()

    def test_toggle_turn(self):
        self.game.toggle_turn()
        self.assertEqual(self.game._turn, 2)

        self.game.toggle_turn()
        self.assertEqual(self.game._turn, 1)

    def test_move(self):
        move1 = self.game.move(P1, 3)
        self.game.toggle_turn()
        move2 = self.game.move(P2, 3)
        self.game.toggle_turn()

        self.assertTrue(move1)
        self.assertTrue(move2)

        np.testing.assert_equal(self.game._board._board, np.array([[0, 0, 0, 1, 0, 0, 0],
                                                                 [0, 0, 0, 2, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0, 0, 0]]))
        
    def test_check_game_end(self):
        self.game.move(P1, 3)
        self.game.toggle_turn()
        self.game.move(P2, 3)
        self.game.toggle_turn()
        self.game.move(P1, 3)
        self.game.toggle_turn()
        self.game.move(P2, 4)
        self.game.toggle_turn()
        self.game.move(P1, 2)
        self.assertFalse(self.game.check_game_end())
        self.game.toggle_turn()

        self.game.move(P2, 2)
        self.game.toggle_turn()
        self.game.move(P1, 1)
        self.game.toggle_turn()
        self.game.move(P2, 1)
        self.assertFalse(self.game.check_game_end())
        self.game.toggle_turn()

        self.game.move(P1, 0)
        self.assertTrue(self.game.check_game_end())
