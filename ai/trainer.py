"""Trains a neural network to play Connect 4 using Deep Q-Learning."""

import pathlib
import math
import random
import time

import numpy as np
import keras

from ai.agent import Agent
from c4.game import Game
from c4.constants import *


class Trainer:
    """Deep Q-Learning AI trainer."""

    def __init__(self, new_model=False):
        """Trainer initializer.

        Args:
            new_model (bool, optional): Whether to create a new model instead of loading a trained one. 
                                        Defaults to False.
        """
        if not new_model and pathlib.Path(MODEL).is_file(): # load already trained model
            self._q_network = keras.models.load_model(MODEL)
            self._target_q_network = keras.models.clone_model(self._q_network)
            self._target_q_network.set_weights(self._q_network.get_weights())
            _, _, self._optimizer = Agent.build_q_networks()
        else:
            self._q_network, self._target_q_network, self._optimizer = Agent.build_q_networks()

        self._game = Game()
        self._board = self._game.board()
        self._agent1 = Agent(P1, self._game, self._q_network, self._target_q_network, self._optimizer)
        self._agent2 = Agent(P2, self._game, self._q_network, self._target_q_network, self._optimizer)

    def _new_epsilon(self, epsilon):
        """Returns a new, lower epsilon value to use for the ε-greedy policy for choosing actions.

        Args:
            epsilon (float): The current epsilon value.

        Returns:
            float: A new epsilon value.
        """
        return max(E_MIN, E_DECAY * epsilon)
    
    def _update_target_network(self, q_network, target_q_network):
        """Updates the target Q-network's weights with a tiny fraction of the Q-network's weights.

        Args:
            q_network (Sequential): Q-network.
            target_q_network (Sequential): Target Q-network.
        """
        for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

    def train(self):
        """Trains the Connect 4 deep learning model."""
        start = time.time()
        episodes = 40000
        epsilon = 1.0

        print("Training...")

        for _ in range(episodes):
            # Reset game, get initial state
            self._game.reset(self._agent1.get_player())

            done = False
            while not done:
                # Agent 1 move
                done = self._agent1.step(epsilon)
                self._game.toggle_turn()

                # Agent 2 move
                if not done:
                    done = self._agent2.step(epsilon)
                    self._game.toggle_turn()

            if self._agent1.num_experiences() >= MINIBATCH_SIZE and self._agent2.num_experiences() >= MINIBATCH_SIZE:
                agent1_exp = self._agent1.sample_experiences()
                agent2_exp = self._agent2.sample_experiences() 
                self._agent1.learn(agent1_exp, GAMMA)
                self._agent2.learn(agent2_exp, GAMMA)
                self._update_target_network(self._q_network, self._target_q_network)

            epsilon = self._new_epsilon(epsilon)

        try:
            keras.models.save_model(self._q_network, MODEL)
        except Exception as e:
            print(e)

        tot_time = time.time() - start
        print(f"\nTraining Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

    def simulate_games_vs_random(self, num_games):
        """Simulates games between the trained AI and random moves.

        Args:
            num_games (int): The number of games to simulate.
        """
        model = keras.models.load_model(MODEL)
        ai_player = P1
        random_player = P2

        wins = 0
        losses = 0
        draws = 0

        print(f"Simulating {num_games} games of AI vs random...")

        for _ in range(num_games):
            self._game.reset(P1)

            game_state = ONGOING
            while game_state == ONGOING: # game loop
                player = self._game.turn()

                if player == ai_player:
                    q_values = model(np.append(self._board.array(), ai_player).reshape(1, -1)).numpy()
                    full_cols = self._board.get_full_cols()
                    np.put(q_values, full_cols, [-math.inf for _ in full_cols])
                    ai_move = np.argmax(q_values)
                    self._game.move(ai_player, ai_move)
                else:
                    self._game.move(random_player, random.choice(self._game.valid_moves()))

                game_state = self._game.check_game_end()
                self._game.toggle_turn()

            if game_state == ai_player:
                wins += 1
            elif game_state == random_player:
                losses += 1
            elif game_state == DRAW:
                draws += 1

            # Switch who goes first
            temp = ai_player
            ai_player = random_player
            random_player = temp

        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Win percentage: {100 * (wins / (wins + losses + draws)):0.2f}%")
