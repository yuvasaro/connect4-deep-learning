"""The Connect 4 game."""

from c4.board import Board
from c4.constants import N, P1, P2, DRAW, ONGOING


class Game:
    """Connect 4 game class."""
    
    def __init__(self):
        "Creates a Connect 4 game instance."
        self._board = Board()
        self._turn = P1 # Player 1's turn

    def turn(self):
        """Returns whose turn it is.

        Returns:
            int: Player 1 or 2.
        """
        return self._turn

    def toggle_turn(self):
        """Toggles whose turn it is."""
        if self._turn == P1:
            self._turn = P2
        else:
            self._turn = P1

    def move(self, player, column):
        """Places a coin on the board for the given player.

        Args:
            player (int): Player 1 or 2.
            column (int): The column to place the coin in.

        Returns:
            bool: Whether the move was successful.
        """
        if self._turn != player: # Safety measure
            return False
        
        if column < 0 or column >= N:
            return False

        for coords in self._board.valid_moves():
            if coords[1] == column:
                self._board.place_coin(player, coords)
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
        if self._board.check_win(self._turn):
            return self._turn
        
        if self._board.check_board_full():
            return DRAW
        
        return ONGOING
    
    def print_board(self):
        """Prints the game board."""
        self._board.print()
