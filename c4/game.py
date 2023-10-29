"""The Connect 4 game."""

from c4.board import Board
from c4.constants import P1, P2, TIE, ONGOING


class Game:
    """Connect 4 game class."""
    
    def __init__(self):
        "Creates a Connect 4 game instance."
        self.board = Board()
        self.turn = P1 # Player 1's turn

    def toggle_turn(self):
        """Toggles whose turn it is."""
        if self.turn == P1:
            self.turn = P2
        else:
            self.turn = P1

    def move(self, player, column):
        """Places a coin on the board for the given player.

        Args:
            player (int): Player 1 or 2.
            column (int): The column to place the coin in.

        Returns:
            bool: Whether the move was successful.
        """
        if self.turn != player: # Safety measure
            return False

        for coords in self.board.valid_moves():
            if coords[1] == column:
                self.board.place_coin(player, coords)
                return True
        return False
    
    def check_game_end(self):
        """Checks if the game has ended.
        
        Returns:
            int: Whether the game ended or not and how it ended.
                0 = game did not end
                1 = player 1 won
                2 = player 2 won
                3 = tie
        """
        # Only the last player who moved can win
        if self.board.check_win(self.turn):
            return self.turn
        
        if self.board.check_board_full():
            return TIE
        
        return ONGOING
