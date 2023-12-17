"""Connect 4 AI agent.

Credit for a lot of the code in this file goes to:
  Stanford and DeepLearning.AI's Machine Learning Specialization (Coursera)
  Course 3, Week 3, Lab: Reinforcement Learning
"""

from collections import namedtuple, deque
import random
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Conv2D, Flatten, LeakyReLU
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import MSE
from scipy.signal import convolve2d

from c4.constants import *

# Use a namedtuple type to represent the agent's experiences
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class Agent:
    """Connect 4 AI agent."""

    @staticmethod
    def build_q_networks():
        """Builds neural networks for Deep Q-Learning.

        Returns:
            tuple: Q-network, target Q-network, optimizer.
        """
        q_network = Sequential([
            Conv2D(64, 4, input_shape=INPUT_SHAPE[1:]),
            LeakyReLU(),
            Conv2D(64, 2, input_shape=INPUT_SHAPE[1:]),
            LeakyReLU(),
            Flatten(),
            Dense(64),
            LeakyReLU(),
            Dense(64),
            LeakyReLU(),
            Dense(N)
        ])

        target_q_network = Sequential([
            Conv2D(64, 4, input_shape=INPUT_SHAPE[1:]),
            LeakyReLU(),
            Conv2D(64, 2, input_shape=INPUT_SHAPE[1:]),
            LeakyReLU(),
            Flatten(),
            Dense(64),
            LeakyReLU(),
            Dense(64),
            LeakyReLU(),
            Dense(N)
        ])

        target_q_network.set_weights(q_network.get_weights()) # start off with same weights

        optimizer = adam_v2.Adam(learning_rate=ALPHA)
        q_network.compile(optimizer=optimizer, loss=MSE)
        target_q_network.compile(optimizer=optimizer, loss=MSE)

        return q_network, target_q_network, optimizer

    def __init__(self, player, game, q_network, target_q_network, optimizer):
        """Agent initializer.

        Args:
            player (int): The player the agent represents (P1 or P2).
            game (Game): A connect 4 game instance.
            q_network (Sequential): Shared Q-network between agents.
            target_q_network (Sequential): Shared target Q-network between agents.
            optimizer (Adam): Adam optimizer used in the learning algorithm.
        """
        self.player = player
        self.game = game
        self.board = game.board
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.optimizer = optimizer
        self.memory_buffer = deque(maxlen=MEMORY_SIZE)

    def get_player(self):
        """Gets the player the agent is playing.

        Returns:
            int: The player the agent is playing.
        """
        return self.player
    
    def set_player(self, player):
        """Sets the player the agent is playing

        Args:
            player (int): The new player the agent is playing.
        """
        self.player = player

    def compute_loss(self, experiences, gamma):
        """Calculates the loss of the current guess of the Q-function.

        Args:
            experiences (tuple): Tuple of Experiences 
                                (namedtuples with fields ["state", "action", "reward", "next_state", "done"]).
            gamma (float): The discount factor.

        Returns:
            TensorFlow Tensor(shape=(0,), dtype=int32): The Mean-Squared Error between the targets and Q-values.
        """
        # 'x' values are states and actions, 'y' values are rewards + max of Q of next states
        states, actions, rewards, next_states, done_vals = experiences

        # The Bellman Equation calculates the maximum return for taking an action at a state and then behaving 
        # optimally after that. The "behaving optimally" is estimated by the target Q-network, and we want to
        # train the Q-network to map a state action pair, (s, a), to the maximum return calculated by the target
        # Q-network, R(s) + gamma * max(Q(s', a')). The target Q-network is updated over time by small fractions
        # of the trained and retrained Q-network so that the target Q-network gets closer and closer to an
        # optimal Q-function.

        # Calculate target values using target Q-network
        max_Q = tf.reduce_max(self.target_q_network(tf.reshape(next_states, BATCH_SHAPE)), axis=1)
        y_targets = rewards + gamma * max_Q * (1 - done_vals) # right side of bellman equation

        # Get the q_values for all N actions done at all the initial states; shape = (len(states), N)
        q_values = self.q_network(tf.reshape(states, BATCH_SHAPE))
        # Get the q_values for the specific action taken at each of the initial states; shape = (len(states), 1)
        #   tf.gather_nd: gather values of parameters (q_values) based on indices
        #   tf.stack: creates a matrix of indices where each row is of the form [row index, action index]
        #     tf.range generates row indices (0 to len(states)-1), tf.casts the action indices to tf int32s (0 to N-1)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

        # Now we can calculate the mean-squared error loss between the Q-values and target values
        return MSE(y_targets, q_values)

    def choose_action(self, q_values, full_cols, epsilon=0):
        """Chooses an action to take using an ε-greedy policy.

        Args:
            q_values (np.ndarray): The calculated Q-values for a state.
            full_cols (list): A list of full columns on the board.
            epsilon (float, optional): Epsilon. Defaults to 0.

        Returns:
            int: The action to take.
        """
        np.put(q_values, full_cols, [-math.inf for _ in full_cols])
        if random.random() > epsilon:
            return np.argmax(q_values[0])
        else:
            return random.choice(self.game.valid_moves())
        
    def get_reward(self, game_state):
        """Returns the reward for the given game board.

        Args:
            game_state (int): The game state.

        Returns:
            int: The reward for the given game board.
        """
        if self.player == P1:
            opp = P2
        else:
            opp = P1

        if game_state == self.player:
            return 1
        elif game_state == DRAW:
            return 0.5
        elif game_state == opp:
            return -1
        else:
            return 0
        
    def step(self, epsilon, playing_opponent=False):
        """Performs one timestep of the game using the agent.

        Args:
            epsilon (float): Epsilon to choose an action with the ε-greedy policy. Defaults to 0.
            playing_opponent (bool, optional): Whether the agent is playing as the opponent. Defaults to False.

        Returns:
            bool: Whether the game has finished.
        """
        if playing_opponent:
            board = self.board.switch_teams_of_coins()
            if self.player == P1:
                player = P2
            else:
                player = P1
        else:
            board = self.board.array()
            player = self.player
            
        state = np.copy(board)
        q_values = self.q_network(state.reshape(INPUT_SHAPE)).numpy()
        full_cols = self.board.get_full_cols()
        action = self.choose_action(q_values, full_cols, epsilon)

        self.game.move(player, action)
        game_state = self.game.check_game_end()
        if playing_opponent:
            next_state = self.board.switch_teams_of_coins()
        else:
            next_state = np.copy(self.board.array())
        reward = self.get_reward(game_state)
        done = (game_state != ONGOING)

        self.memory_buffer.append(Experience(state, action, reward, next_state, done))

        return done
    
    @tf.function
    def learn(self, experiences, gamma):
        """Runs a gradient descent step and updates the Q-network's and target Q-network's weights.

        Args:
            gamma (float): The discount factor.
            experiences (tuple): Tuple of Tensors representing fields ["state", "action", "reward", "next_state", "done"].
        """
        # Let TensorFlow see how the loss was calculated
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, gamma)
        
        # Calculate gradients with respect to the Q-network's weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Update the Q-network's weights (gradient descent: w = w - alpha * gradient)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

    def num_experiences(self):
        """Returns the agent's current number of experiences.

        Returns:
            int: Number of experiences in the agent's memory buffer.
        """
        return len(self.memory_buffer)
    
    def sample_experiences(self):
        """Gets a sample of experiences of size MINIBATCH_SIZE from the memory buffer.

        Returns:
            tuple: Tuple of TensorFlow Tensors of length MINIBATCH_SIZE: states, actions, rewards, next_states, done_vals.
        """
        experiences = random.sample(self.memory_buffer, k=BATCH_SIZE)
        states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
        done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)
        return (states, actions, rewards, next_states, done_vals)
