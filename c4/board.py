"""A Connect 4 Board class."""

import numpy as np
from scipy.signal import convolve2d

from c4.constants import *


class Board:
    """The Connect 4 game board."""
    
    def __init__(self):
        """Creates a Board instance."""
        self._board = np.zeros((M, N), dtype=int)

    def array(self):
        """Gets the M by N numpy board array.

        Returns:
            np.ndarray: The board array.
        """
        return self._board

    def valid_moves(self):
        """Gets a list of all valid moves as tuples.
        
        Returns:
            list: A list of tuples containing the possible moves.
        """
        moves = []

        for i in range(M):
            for j in range(N):

                if self._board[i, j] == SPACE:

                    # Either bottom row or there's a coin below
                    if i == 0 or self._board[i - 1, j] != SPACE:
                        moves.append((i, j))

        return moves
    
    def place_coin(self, player, coords):
        """Places a player's coin on the board.

        Args:
            player (int): Player 1 or 2.
            coords (tuple): Coordinates to place the coin on the board.
        """
        self._board[coords[0], coords[1]] = player

    def get_coin(self, coords):
        """Returns the coin at the given coordinates.

        Args:
            coords (tuple): Coordinates on the board

        Returns:
            int: The coin on the given coordinates
        """
        if (coords[0] < 0 or coords[0] >= M or coords[1] < 0 or coords[1] >= N):
            return -1
        return self._board[coords[0], coords[1]]

    def check_win(self, player):
        """Checks whether the given player has a sequence of 4 coins in a row.

        Args:
            player (int): The player to check for a win.

        Returns:
            bool: Whether the player won or not.
        """
        # Win check matrices: horizontal, vertical, diagonal
        horizontal_kernel = np.full((1, 4), 1)
        vertical_kernel = np.transpose(horizontal_kernel)
        neg_diag_kernel = np.eye(4, dtype=int)
        pos_diag_kernel = np.fliplr(neg_diag_kernel)

        win_patterns = [horizontal_kernel, vertical_kernel, neg_diag_kernel, pos_diag_kernel]

        for kernel in win_patterns:
            # Use convolve2d on player specific move board to check for 4 in a row
            if (convolve2d(self._board == player, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def check_board_full(self):
        """Returns whether there are any spaces left in the board.

        Returns:
            bool: Whether the board is full.
        """
        return not (SPACE in self._board)
    
    def get_full_cols(self):
        """Returns a list of full columns in the board.

        Returns:
            list: A list of full columns in the board.
        """
        full_cols = []
        for j in range(N):
            if SPACE not in self._board[:, j]:
                full_cols.append(j)
        return full_cols
    
    def reset(self):
        """Resets the game board."""
        self._board.fill(SPACE)
    
    def print(self):
        """Prints the Connect 4 board to the console."""
        print(np.flipud(self._board))

    def switch_teams_of_coins(self):
        """Switches the teams of the coins on the board.

        Returns:
            np.ndarray: A copy of the board with the coin teams switched.
        """
        board = np.copy(self._board)
        board[board == P1] = PLACEHOLDER
        board[board == P2] = P1
        board[board == PLACEHOLDER] = P2
        return board
