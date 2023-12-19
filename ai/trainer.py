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
        if not new_model and pathlib.Path(MODEL_P1).is_file() and pathlib.Path(MODEL_P2).is_file(): # load already trained model
            self.q_network_p1 = keras.models.load_model(MODEL_P1)
            self.target_q_network_p1 = keras.models.clone_model(self.q_network_p1)
            self.target_q_network_p1.set_weights(self.q_network_p1.get_weights())

            self.q_network_p2 = keras.models.load_model(MODEL_P2)
            self.target_q_network_p2 = keras.models.clone_model(self.q_network_p2)
            self.target_q_network_p2.set_weights(self.q_network_p2.get_weights())

            _, _, self.optimizer = Agent.build_q_networks()
        else:
            self.q_network_p1, self.target_q_network_p1, self.optimizer = Agent.build_q_networks()
            self.q_network_p2, self.target_q_network_p2, _ = Agent.build_q_networks()

        self.game = Game()
        self.board = self.game.board
        self.agent_p1 = Agent(P1, self.game, self.q_network_p1, self.target_q_network_p1, self.optimizer)
        self.agent_p2 = Agent(P2, self.game, self.q_network_p2, self.target_q_network_p2, self.optimizer)

    def new_epsilon(self, epsilon):
        """Returns a new, lower epsilon value to use for the ε-greedy policy for choosing actions.

        Args:
            epsilon (float): The current epsilon value.

        Returns:
            float: A new epsilon value.
        """
        return max(E_MIN, E_DECAY * epsilon)
    
    def update_target_network(self, q_network, target_q_network):
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
        episodes = 10000
        epsilon = 1.0

        print("Training...")

        for i in range(1, episodes + 1):
            # Reset game, get initial state
            self.game.reset(self.agent_p1.get_player())

            done = False
            while not done:
                # Agent 1 move
                done = self.agent_p1.step(epsilon)
                self.game.toggle_turn()

                # Agent 2 move
                if not done:
                    done = self.agent_p2.step(epsilon)
                    self.game.toggle_turn()

            if self.agent_p1.num_experiences() >= BATCH_SIZE:
                agent1_exp = self.agent_p1.sample_experiences()
                self.agent_p1.learn(agent1_exp, GAMMA)
            self.update_target_network(self.q_network_p1, self.target_q_network_p1)

            if self.agent_p2.num_experiences() >= BATCH_SIZE:
                agent2_exp = self.agent_p2.sample_experiences()
                self.agent_p2.learn(agent2_exp, GAMMA)
            self.update_target_network(self.q_network_p2, self.target_q_network_p2)

            epsilon = self.new_epsilon(epsilon)

        try:
            keras.models.save_model(self.q_network_p1, MODEL_P1)
            keras.models.save_model(self.q_network_p2, MODEL_P2)
        except Exception as e:
            print(e)

        tot_time = time.time() - start
        print(f"\nTraining Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

    def simulate_games_vs_random(self, num_games):
        """Simulates games between the trained AI and random moves.

        Args:
            num_games (int): The number of games to simulate.
        """
        model_p1 = keras.models.load_model(MODEL_P1)
        model_p2 = keras.models.load_model(MODEL_P2)
        ai_player = P1
        random_player = P2

        wins = 0
        losses = 0
        draws = 0

        print(f"Simulating {num_games} games of AI vs random...")

        for _ in range(num_games):
            self.game.reset(P1)

            game_state = ONGOING
            while game_state == ONGOING: # game loop
                player = self.game.turn

                if player == ai_player:
                    if ai_player == P1:
                        q_values = model_p1(self.board.array().reshape(INPUT_SHAPE)).numpy()
                    else: # P2
                        q_values = model_p2(self.board.array().reshape(INPUT_SHAPE)).numpy()

                    full_cols = self.board.get_full_cols()
                    np.put(q_values, full_cols, [-math.inf for _ in full_cols])
                    ai_move = np.argmax(q_values)
                    self.game.move(ai_player, ai_move)
                else:
                    self.game.move(random_player, random.choice(self.game.valid_moves()))

                game_state = self.game.check_game_end()
                self.game.toggle_turn()

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
