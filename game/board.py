"""A Connect 4 Board class."""

import numpy as np
from scipy.signal import convolve2d


class Board:
    """The Connect 4 game board."""

    # m x n matrix
    m = 6
    n = 7
    
    def __init__(self):
        """Creates a Board instance."""
        self.board = np.zeros((self.m, self.n))

        # Store moves of each player separately
        self.p1_moves = np.zeros((self.m, self.n))
        self.p2_moves = np.zeros((self.m, self.n))

    def valid_moves(self):
        """Gets a list of all valid moves as tuples.
        
        Returns:
            list: A list of tuples containing the possible moves.
        """
        moves = []

        for i in range(self.m):
            for j in range(self.n):

                if self.board[i, j] == 0:

                    # Either bottom row or there's a coin below
                    if i == 0 or self.board[i - 1, j] != 0:
                        moves.append((i, j))

        return moves
    
    def place_coin(self, player, coords):
        """Places a player's coin on the board.

        Args:
            player (int): Player 1 or 2.
            coords (tuple): Coordinates to place the coin on the board.
        """
        self.board[coords[0], coords[1]] = player

        # Store 1 for player move and 0 for other spots for win check purposes
        if player == 1:
            self.p1_moves[coords[0], coords[1]] = 1
        else:
            self.p2_moves[coords[0], coords[1]] = 1

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
        pos_diag_kernel = np.eye(4, dtype=int)
        neg_diag_kernel = np.fliplr(pos_diag_kernel)

        win_patterns = [horizontal_kernel, vertical_kernel, pos_diag_kernel, neg_diag_kernel]

        if player == 1:
            check_board = self.p1_moves
        else:
            check_board = self.p2_moves

        for kernel in win_patterns:
            # Use convolve2d on player specific move board
            if (convolve2d(check_board, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def check_board_full(self):
        """Returns whether there are any spaces left in the board.

        Returns:
            bool: Whether the board is full.
        """
        return not (0 in self.board)
    
    def print(self):
        """Prints the Connect 4 board to the console."""
        print(np.flipud(self.board))
