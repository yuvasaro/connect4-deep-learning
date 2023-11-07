"""A Connect 4 Board class."""

import numpy as np
from scipy.signal import convolve2d

from c4.constants import M, N, P1, SPACE


class Board:
    """The Connect 4 game board."""
    
    def __init__(self):
        """Creates a Board instance."""
        self._board = np.zeros((M, N), dtype=int)

        # Store moves of each player separately
        self._p1_moves = np.zeros((M, N), dtype=int)
        self._p2_moves = np.zeros((M, N), dtype=int)

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

        # Store 1 for player move and 0 for other spots for win check purposes
        if player == P1:
            self._p1_moves[coords[0], coords[1]] = 1
        else:
            self._p2_moves[coords[0], coords[1]] = 1

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

        if player == P1:
            check_board = self._p1_moves
        else:
            check_board = self._p2_moves

        for kernel in win_patterns:
            # Use convolve2d on player specific move board to check for 4 in a row
            if (convolve2d(check_board, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def check_board_full(self):
        """Returns whether there are any spaces left in the board.

        Returns:
            bool: Whether the board is full.
        """
        return not (SPACE in self._board)
    
    def print(self):
        """Prints the Connect 4 board to the console."""
        print(np.flipud(self._board))
