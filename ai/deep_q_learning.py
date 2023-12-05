"""
Neural networks for Connect 4 deep learning.

Credit for the code in this file goes to:
  Stanford and DeepLearning.AI's Machine Learning Specialization (Coursera)
  Course 3, Week 3, Lab: Reinforcement Learning
"""

from collections import namedtuple, deque
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import mse

from ai.c4_gym_env import Connect4Env
from c4.constants import *


class DeepQLearning:
    """A Connect 4 Deep Q-Learning class."""

    # Using a named tuple for the model's experiences, represented as (s, a, R(s), s')
    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self):
        """DeepQLearning initializer."""
        self.env = Connect4Env(P1, render_mode="human")

    def _build_q_networks(self):
        """Builds neural networks for Deep Q-Learning.

        Returns:
            tuple: Q-network, target Q-network, optimizer.
        """
        q_network = Sequential([
            Input(STATE_SIZE),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(N, activation="linear")
        ])

        target_q_network = Sequential([
            Input(STATE_SIZE),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(N, activation="linear")
        ])

        optimizer = adam_v2(learning_rate=ALPHA)

        return q_network, target_q_network, optimizer

    def _compute_loss(self, experiences, gamma, q_network, target_q_network):
        """Calculates the loss of the current guess of the Q-function.

        Args:
            experiences (tuple): Tuple of Experiences 
                                (namedtuples with fields ["state", "action", "reward", "next_state", "done"]).
            gamma (float): The discount factor.
            q_network (Sequential): Q-network.
            target_q_network (Sequential): Target Q-network.

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
        max_Q = tf.reduce_max(target_q_network(next_states), axis=1)
        y_targets = rewards + gamma * max_Q * (1 - done_vals) # right side of bellman equation

        # Get the q_values for all N actions done at all the initial states; shape = (len(states), N)
        q_values = q_network(states)
        # Get the q_values for the specific action taken at each of the initial states; shape = (len(states), 1)
        #   tf.gather_nd: gather values of parameters (q_values) based on indices
        #   tf.stack: creates a matrix of indices where each row is of the form [row index, action index]
        #     tf.range generates row indices (0 to len(states)-1), tf.casts the action indices to tf int32s (0 to N-1)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

        # Now we can calculate the mean-squared error loss between the Q-values and target values
        return mse(y_targets, q_values)

    def _update_target_network(self, q_network, target_q_network):
        """Updates the target Q-network's weights with a tiny fraction of the Q-network's weights.

        Args:
            q_network (Sequential): Q-network.
            target_q_network (Sequential): Target Q-network.
        """
        for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

    def _learn_step(self, experiences, gamma, q_network, target_q_network, optimizer):
        """Runs a gradient descent step and updates the Q-network's and target Q-network's weights.

        Args:
            experiences (tuple): Tuple of Experiences 
                                (namedtuples with fields ["state", "action", "reward", "next_state", "done"]).
            gamma (float): The discount factor.
            q_network (Sequential): Q-network.
            target_q_network (Sequential): Target Q-network.
            optimizer (adam_v2): Adam optimizer from TensorFlow.
        """
        # Let TensorFlow see how the loss was calculated
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences, gamma, q_network, target_q_network)
        
        # Calculate gradients with respect to the Q-network's weights
        gradients = tape.gradient(loss, q_network.trainable_variables)
        
        # Update the Q-network's weights (gradient descent: w = w - alpha * gradient)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

        # Update the target Q-network's weights with a tiny fraction of the Q-network's weights
        self._update_target_network(q_network, target_q_network)

    def _choose_action(self, q_values, epsilon=0):
        """Chooses an action to take using an ε-greedy policy.

        Args:
            q_values (np.ndarray): The calculated Q-values for a state.
            epsilon (float, optional): Epsilon. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if random.random() > epsilon:
            return np.argmax(q_values.numpy()[0])
        else:
            return random.choice(np.arange(N))
    
    def _check_update_conditions(self, t, C, memory_buffer):
        """Do a learning update every C timesteps, and if there are enough experiences in the memory buffer.

        Args:
            t (int): Current number of timesteps.
            C (int): Learning update interval.
            memory_buffer (deque): Memory buffer containing experience tuples.

        Returns:
            bool: Whether a learning update will be done.
        """
        return (t + 1) % C == 0 and len(memory_buffer) > MINIBATCH_SIZE
    
    def _sample_experiences(self, memory_buffer):
        """Gets a sample of experiences of size MINIBATCH_SIZE from the memory buffer.

        Args:
            memory_buffer (deque): Memory buffer containing experience tuples.

        Returns:
            tuple: Tuple of TensorFlow Tensors of length MINIBATCH_SIZE: states, actions, rewards, next_states, done_vals.
        """
        experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
        states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
        done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)
        return (states, actions, rewards, next_states, done_vals)
    
    def _new_epsilon(self, epsilon):
        """Returns a new, lower epsilon value to use for the ε-greedy policy for choosing actions.

        Args:
            epsilon (float): The current epsilon value.

        Returns:
            float: A new epsilon value.
        """
        return max(E_MIN, E_DECAY * epsilon)

    def train(self):
        """Trains the Connect 4 deep learning model."""
        start = time.time()

        episodes = 2000
        max_timesteps = 1000

        total_point_history = []
        num_points_to_average = 100
        epsilon = 1.0

        # Initialize a deque to store experiences
        memory_buffer = deque(maxlen=MEMORY_SIZE)

        # Initialize networks; start with equal weights on Q-network and target Q-network
        q_network, target_q_network, optimizer = self._build_q_networks()
        target_q_network.set_weights(q_network.get_weights())

        for i in range(episodes):
            # Reset game, get initial state
            state = self.env.reset()
            total_points = 0

            for t in range(max_timesteps):
                # At state s, choose action a using an ε-greedy policy
                state_qn = np.expand_dims(state, axis=0)
                q_values = q_network(state_qn)
                action = self._choose_action(q_values, epsilon)

                # Take action a, get reward R(s), end up at next state s'
                next_state, reward, done, _, _ = self.env.step(action)

                # Store this experience in the memory buffer
                memory_buffer.append(self.Experience(state, action, reward, next_state, done))

                if self._check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
                    # Sample experiences from the memory buffer and perform a learning update
                    experiences = self._sample_experiences(memory_buffer)
                    self._learn_step(experiences, GAMMA, q_network, target_q_network, optimizer)
                
                state = next_state.copy()
                total_points += reward

                if done:
                    break
            
            total_point_history.append(total_points)
            latest_average = np.mean(total_point_history[-num_points_to_average:])

            epsilon = self._new_epsilon(epsilon)

            print(f"\rEpisode {i+1}: Average points of latest {num_points_to_average} episodes: {latest_average:.2f}", end="")

            if (i+1) % num_points_to_average == 0:
                print(f"\rEpisode {i+1}: Average points of latest {num_points_to_average} episodes: {latest_average:.2f}")

            # Consider the environment solved if the latest average points is at least 200
            if latest_average >= 200.0:
                print(f"\n\nEnvironment solved in {i+1} episodes!")
                q_network.save("connect4_model.h5")
                break

        tot_time = time.time() - start
        print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
