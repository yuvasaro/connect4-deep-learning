"""A Connect 4 gym environment."""

import gym
from gym import spaces
import numpy as np
import pygame

from c4.game import Game
from c4.constants import *


class Connect4Env(gym.Env):
    """A Connect 4 gym environment."""

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

    # Colors
    blue = (0, 0, 255)
    black = (0, 0, 0)
    red = (255, 0, 0) # P1
    yellow = (255, 255, 0) # P2

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, player, render_mode=None):
        """Connect4Env initializer.

        Args:
            player (int): The player the AI is playing.
            render_mode (str, optional): 'human' or 'rgb_array'. Defaults to None.
        """

        # Observation space contains the locations of all the pieces on the board
        # Storing all played moves in separate spaces for each player
        self.observation_space = spaces.Dict(
            {
                "p1_moves": spaces.Box(np.array([0, 0]), np.array([M, N])),
                "p2_moves": spaces.Box(np.array([0, 0]), np.array([M, N]))
            }
        )

        # Actions are the columns to place pieces in
        self.action_space = spaces.Discrete(N)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.player = player

    def _get_obs(self):
        """Returns an observation of the current game board.

        Returns:
            dict: Observation of the current game board.
        """
        return {
            "p1_moves": self.board.get_p1_moves(),
            "p2_moves": self.board.get_p2_moves()
        }

    def _get_info(self):
        """Returns a dictionary containing the game state.

        Returns:
            dict: A dictionary containing the game state.
        """
        return {"game_state": self.game_state}
    
    def _get_reward(self):
        return 1 if self.game_state == self.player else 0

    def reset(self):
        """Resets the game.

        Returns:
            tuple: Observation and info of the resetted game.
        """
        old_game = self.game
        self.game = Game()
        del old_game
        
        self.board = self.game.board()
        self.game_state = ONGOING

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """Runs one timestep of the environment.

        Args:
            action (int): The column to place a coin in.

        Returns:
            tuple: Observation (dict), reward (float), done (bool), False, info (dict)
        """
        self.game.move(self.player, action)
        self.game_state = self.game.check_game_end()

        done = self.game_state != ONGOING
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.game.toggle_turn() # opponent's turn
        
        return observation, reward, done, False, info
    
    def render(self):
        """Returns an RGB array of the game board if the render mode is set to 'rgb_array'.

        Returns:
            np.ndarray: RGB array of the game board.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        """Renders a single frame of the Connect 4 game.

        Returns:
            np.ndarray: RGB array of the game board if the render mode is 'rgb_array'.
        """

        # Setup pygame window and clock
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Connect 4")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw the Connect 4 board
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

        if self.game_state != ONGOING:
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
    
        if self.render_mode == "human":
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        else: # rgb array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window), axes=(1, 0, 2))
            )
        
    def close(self):
        """Closes the game window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
