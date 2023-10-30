"""Console Connect 4 game."""

import sys

from c4.constants import P1, P2, DRAW, ONGOING
from c4.game import Game
from gui.c4_gui import C4_GUI


def main():
    print("Welcome to Connect 4!\n")

    game = Game()
    game.print_board()

    game_state = ONGOING
    while game_state == ONGOING: # game loop
        player = game.turn()

        valid_move = False
        while not valid_move:
            try:
                player_move = int(input(f"\nPlayer {player}, enter move (1-7): ")) - 1
            except:
                continue
            valid_move = game.move(player, player_move)

        print()
        game.print_board()

        game_state = game.check_game_end()
        game.toggle_turn()

    if game_state == P1 or game_state == P2:
        print(f"\nPlayer {game_state} wins!")
    elif game_state == DRAW:
        print("\nIt's a draw!")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "console":
        main()
    else:
        C4_GUI()
