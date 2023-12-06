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
from tensorflow.python.keras.layers import InputLayer, Dense
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
        input_size = M * N + 1 # state is all board positions and player number

        q_network = Sequential([
            InputLayer((input_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(N, activation="linear")
        ])

        target_q_network = Sequential([
            InputLayer((input_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(N, activation="linear")
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
        self._player = player
        self._game = game
        self._board = game.board()
        self._q_network = q_network
        self._target_q_network = target_q_network
        self._optimizer = optimizer

        self._memory_buffer = deque(maxlen=MEMORY_SIZE)
        self._moves_made = 0

    def get_player(self):
        """Gets the player the agent is playing.

        Returns:
            int: The player the agent is playing.
        """
        return self._player

    def _compute_loss(self, experiences, gamma):
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
        max_Q = tf.reduce_max(self._target_q_network(next_states), axis=1)
        y_targets = rewards + gamma * max_Q * (1 - done_vals) # right side of bellman equation

        # Get the q_values for all N actions done at all the initial states; shape = (len(states), N)
        q_values = self._q_network(states)
        # Get the q_values for the specific action taken at each of the initial states; shape = (len(states), 1)
        #   tf.gather_nd: gather values of parameters (q_values) based on indices
        #   tf.stack: creates a matrix of indices where each row is of the form [row index, action index]
        #     tf.range generates row indices (0 to len(states)-1), tf.casts the action indices to tf int32s (0 to N-1)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

        # Now we can calculate the mean-squared error loss between the Q-values and target values
        return MSE(y_targets, q_values)

    def _choose_action(self, q_values, full_cols, epsilon=0):
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
            return random.choice(self._game.valid_moves())
        
    def _get_reward(self):
        """Returns the reward for the current game board.

        Returns:
            int: The reward for the current game board.
        """
        # We want to check how many 2 in a rows, 3 in a rows, and 4 in a rows each player has
        reward_2 = 10
        reward_3 = 40
        reward_4 = 100 # 4 in a row is a win
        reward_num_coins = -10 # want to win faster, so penalize having more coins on the board

        # Create horizontal, vertical, and diagonal kernels for 2, 3, and 4 in a row
        horizontal_2 = np.full((1, 2), 1)
        vertical_2 = np.transpose(horizontal_2)
        neg_diag_2 = np.eye(2, dtype=int)
        pos_diag_2 = np.fliplr(neg_diag_2)
        two_in_a_row = [horizontal_2, vertical_2, neg_diag_2, pos_diag_2]

        horizontal_3 = np.full((1, 3), 1)
        vertical_3 = np.transpose(horizontal_2)
        neg_diag_3 = np.eye(3, dtype=int)
        pos_diag_3 = np.fliplr(neg_diag_2)
        three_in_a_row = [horizontal_3, vertical_3, neg_diag_3, pos_diag_3]

        horizontal_4 = np.full((1, 4), 1)
        vertical_4 = np.transpose(horizontal_2)
        neg_diag_4 = np.eye(4, dtype=int)
        pos_diag_4 = np.fliplr(neg_diag_2)
        four_in_a_row = [horizontal_4, vertical_4, neg_diag_4, pos_diag_4]

        # Positive reward for agent having coins in a row, negative reward for opponent having coins in a row
        agent = self._player
        if agent == P1:
            opp = P2
        else:
            opp = P1

        agent_moves = (self._board.array() == agent)
        opp_moves = (self._board.array() == opp)
        total_reward = 0

        # Twos in a row
        agent_twos = 0
        opp_twos = 0
        for pattern in two_in_a_row:
            agent_twos += np.sum(convolve2d(agent_moves, pattern, mode="valid") == 2)
            opp_twos += np.sum(convolve2d(opp_moves, pattern, mode="valid") == 2)
            total_reward += reward_2 * (agent_twos - opp_twos)

        # Threes in a row
        agent_threes = 0
        opp_threes = 0
        for pattern in three_in_a_row:
            agent_threes += np.sum(convolve2d(agent_moves, pattern, mode="valid") == 3)
            opp_threes += np.sum(convolve2d(opp_moves, pattern, mode="valid") == 3)
            total_reward += reward_3 * (agent_threes - opp_threes)

        # Fours in a row
        agent_fours = 0
        opp_fours = 0
        for pattern in four_in_a_row:
            agent_fours += np.sum(convolve2d(agent_moves, pattern, mode="valid") == 4)
            opp_fours += np.sum(convolve2d(opp_moves, pattern, mode="valid") == 4)
            total_reward += reward_4 * (agent_fours - opp_fours)

        total_reward += reward_num_coins * np.sum(agent_moves)

        return total_reward
        
    def step(self, epsilon):
        """Performs one timestep of the game using the agent.

        Args:
            epsilon (float): Epsilon to choose an action with the ε-greedy policy. Defaults to 0.

        Returns:
            bool: Whether the game has finished.
        """
        state = np.append(self._board.array(), self._player)
        q_values = self._q_network(state.reshape(1, -1)).numpy()
        full_cols = self._board.get_full_cols()
        action = self._choose_action(q_values, full_cols, epsilon)

        self._game.move(self._player, action)
        game_state = self._game.check_game_end()
        next_state = np.append(self._board.array(), self._player)
        reward = self._get_reward()
        done = (game_state != ONGOING)

        self._memory_buffer.append(Experience(state, action, reward, next_state, done))

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
            loss = self._compute_loss(experiences, gamma)
        
        # Calculate gradients with respect to the Q-network's weights
        gradients = tape.gradient(loss, self._q_network.trainable_variables)
        
        # Update the Q-network's weights (gradient descent: w = w - alpha * gradient)
        self._optimizer.apply_gradients(zip(gradients, self._q_network.trainable_variables))

    def num_experiences(self):
        """Returns the agent's current number of experiences.

        Returns:
            int: Number of experiences in the agent's memory buffer.
        """
        return len(self._memory_buffer)
    
    def sample_experiences(self):
        """Gets a sample of experiences of size MINIBATCH_SIZE from the memory buffer.

        Returns:
            tuple: Tuple of TensorFlow Tensors of length MINIBATCH_SIZE: states, actions, rewards, next_states, done_vals.
        """
        experiences = random.sample(self._memory_buffer, k=MINIBATCH_SIZE)
        states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
        done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)
        return (states, actions, rewards, next_states, done_vals)
