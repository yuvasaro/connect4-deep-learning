"""Connect 4 GUI."""

import math

import pygame
pygame.init()
pygame.font.init()

from c4.constants import *
from c4.game import Game


class C4_GUI:
    """Connect 4 GUI."""

    # Game and board objects
    game = Game()
    board = game.board()
    game_state = ONGOING
    can_place = False

    # GUI settings
    slot_size = 100
    circle_radius = 0.4 * slot_size
    width = N * slot_size
    height = (M + 1) * slot_size
    font_size = 60
    font = pygame.font.SysFont('Comic Sans MS', font_size)
    fps = 60

    # Colors
    blue = (0, 0, 255)
    black = (0, 0, 0)
    red = (255, 0, 0) # P1
    yellow = (255, 255, 0) # P2
    
    def __init__(self):
        """Creates a Connect 4 GUI instance."""
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect 4")
        self.clock = pygame.time.Clock()
      
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                self.window.fill(self.black)

                for i in range(N):
                    for j in range(M):
                        pygame.draw.rect(
                            self.window, 
                            self.blue, 
                            (
                                i * self.slot_size, 
                                (j + 1) * self.slot_size, 
                                self.slot_size,
                                self.slot_size
                            )
                        )

                        color = self.black
                        flip_j = M - j - 1
                        if self.board.get_coin((flip_j, i)) == P1:
                            color = self.red
                        elif self.board.get_coin((flip_j, i)) == P2:
                            color = self.yellow

                        pygame.draw.circle(
                            self.window,
                            color,
                            (
                                i * self.slot_size + self.slot_size / 2,
                                (j + 1) * self.slot_size + self.slot_size / 2
                            ),
                            self.circle_radius
                        )
                
                if self.game_state == ONGOING:
                    # Draw coin near cursor
                    if event.type == pygame.MOUSEMOTION:
                        mouse_x = event.pos[0]
                        if self.game.turn() == P1:
                            pygame.draw.circle(self.window, self.red, (mouse_x, self.slot_size / 2), self.circle_radius)
                        else:
                            pygame.draw.circle(self.window, self.yellow, (mouse_x, self.slot_size / 2), self.circle_radius)
                    
                    if event.type == pygame.MOUSEBUTTONUP:
                        self.can_place = True

                    # Place coin in closest column to cursor
                    if event.type == pygame.MOUSEBUTTONDOWN and self.can_place:
                        mouse_x = event.pos[0]
                        col = int(math.floor(mouse_x / self.slot_size))
                        self.game.move(self.game.turn(), col)
                        self.can_place = False

                        self.game_state = self.game.check_game_end()
                        self.game.toggle_turn()
                else:
                    if self.game_state == P1:
                        text = f"Player {self.game_state} wins!"
                        text_color = self.red
                    elif self.game_state == P2:
                        text = f"Player {self.game_state} wins!"
                        text_color = self.yellow
                    else:
                        text = "It's a draw!"
                        text_color = self.blue

                    text_render = self.font.render(text, False, text_color)
                    self.window.blit(text_render, (1.7 * self.slot_size, self.slot_size / 12))

                    if event.type == pygame.KEYDOWN:
                        # Start a new game if the enter key is pressed
                        if pygame.key.get_pressed()[pygame.K_RETURN]:
                            old_game = self.game
                            self.game = Game()
                            del old_game
                            
                            self.board = self.game.board()
                            self.game_state = ONGOING

            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
