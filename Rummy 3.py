import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.callbacks import BaseCallback
import torch

# Assuming your RummyEnv is defined correctly in the same script or imported from 'rummy_env.py'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RummyEnv(gym.Env):
    def __init__(self, max_rounds=100):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)
        self.action_space = spaces.Discrete(108)
        self.rng = np.random.default_rng()
        self.max_rounds = max_rounds  # Maximum number of rounds before declaring a winner
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.draw_pile = self._create_deck()
        self.rng.shuffle(self.draw_pile)
        self.hand_1 = self._draw_initial_hand()
        self.hand_2 = self._draw_initial_hand()
        self.discard_pile = []
        self.current_turn = 1
        self.rounds = 0  # Track rounds

        observation = self._get_observation()
        return observation, {}

    def _create_deck(self):
        suits = ['♥', '♦', '♣', '♠']
        ranks = list(range(1, 14))
        deck = [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2
        return deck

    def _draw_initial_hand(self):
        return [self.draw_pile.pop() for _ in range(13)]

    def _get_observation(self):
        obs = [0] * 108
        for card in self.hand_1:
            obs[self._card_to_index(card)] = 1
        for card in self.hand_2:
            obs[self._card_to_index(card)] = 1
        for card in self.discard_pile:
            obs[54 + self._card_to_index(card)] = 1
        return np.array(obs, dtype=np.float32)

    def _card_to_index(self, card):
        if card == 'joker':
            return 52
        rank, suit = card
        suit_offset = {'♥': 0, '♦': 13, '♣': 26, '♠': 39}[suit]
        return suit_offset + rank - 1

    def step(self, action):
        if action < 54:
            action_type = 0
            card_index = action
        else:
            action_type = 1
            card_index = action - 54

        reward = 0
        terminated = False
        truncated = False
        win = False  # Default win is False

        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        if action_type == 0:
            if 0 <= card_index < len(active_hand):
                discarded_card = active_hand.pop(card_index)
                self.discard_pile.append(discarded_card)
                reward += 1
            else:
                reward -= 5  # Heavier penalty for invalid discard
        elif action_type == 1:
            if len(self.draw_pile) > 0:
                drawn_card = self.draw_pile.pop()
                active_hand.append(drawn_card)
                reward += 0  # Neutral reward for drawing a card
            else:
                terminated = True

        # Check for game-ending conditions
        if len(self.hand_1) == 0:
            terminated = True
            reward += 100  # Player 1 victory
            win = True
        elif len(self.hand_2) == 0:
            terminated = True
            reward += 100  # Player 2 victory
            win = False
        elif self.rounds >= self.max_rounds:  # Max rounds reached, declare winner based on fewer cards
            terminated = True
            win = self._decide_winner_based_on_cards()

        reward += self._evaluate_hand(active_hand)

        self.current_turn = 3 - self.current_turn
        self.rounds += 1  # Increment round count

        observation = self._get_observation()
        info = {'win': win}

        return observation, reward, terminated, truncated, info

    def _evaluate_hand(self, hand):
        # Reward shaping: give bonus for creating sequences
        return 10 * self._count_pure_sequences(hand) + 5 * self._count_impure_sequences(hand) + self._count_sets(hand)

    def _count_pure_sequences(self, hand):
        """Count the number of pure sequences (runs of consecutive cards in the same suit)."""
        sequences = 0
        hand_sorted = sorted(hand, key=lambda x: (x[1], x[0]))  # Sort by suit, then rank
        temp_sequence = []

        for card in hand_sorted:
            if temp_sequence and card[1] == temp_sequence[-1][1] and card[0] == temp_sequence[-1][0] + 1:
                temp_sequence.append(card)
            else:
                if len(temp_sequence) >= 3:
                    sequences += 1
                temp_sequence = [card]

        if len(temp_sequence) >= 3:
            sequences += 1
        return sequences

    def _count_impure_sequences(self, hand):
        """Count the number of impure sequences (runs that can use jokers)."""
        sequences = 0
        hand_sorted = sorted(hand, key=lambda x: (x[1], x[0]))  # Sort by suit, then rank
        temp_sequence = []

        for card in hand_sorted:
            if card != 'joker' and temp_sequence and card[1] == temp_sequence[-1][1] and card[0] == temp_sequence[-1][0] + 1:
                temp_sequence.append(card)
            elif card == 'joker':
                temp_sequence.append(card)
            else:
                if len(temp_sequence) >= 3:
                    sequences += 1
                temp_sequence = [card]

        if len(temp_sequence) >= 3:
            sequences += 1
        return sequences

    def _count_sets(self, hand):
        """Count the number of sets (3+ cards of the same rank)."""
        rank_counts = {}
        for card in hand:
            if card != 'joker':
                rank = card[0]
                if rank not in rank_counts:
                    rank_counts[rank] = 1
                else:
                    rank_counts[rank] += 1

        sets = sum(1 for count in rank_counts.values() if count >= 3)
        return sets

    def _decide_winner_based_on_cards(self):
        # If the max rounds are reached, the player with fewer cards wins
        if len(self.hand_1) < len(self.hand_2):
            return True  # Player 1 wins
        elif len(self.hand_1) > len(self.hand_2):
            return False  # Player 2 wins
        else:
            # In case of equal cards, check sequences
            return self._decide_winner_based_on_sequences()

    def _decide_winner_based_on_sequences(self):
        # Compare the number of sequences each player has formed
        p1_sequences = self._count_pure_sequences(self.hand_1) + self._count_impure_sequences(self.hand_1)
        p2_sequences = self._count_pure_sequences(self.hand_2) + self._count_impure_sequences(self.hand_2)

        if p1_sequences > p2_sequences:
            return True  # Player 1 wins based on more sequences
        elif p2_sequences > p1_sequences:
            return False  # Player 2 wins based on more sequences
        else:
            return None  # If sequences are equal, declare as Draw (should not happen with other checks)

# Reward Logger Callback for collecting rewards over time during training
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps = []  # Store timesteps
        self.rewards = []    # Store rewards

    def _on_step(self) -> bool:
        # Collect reward information and timestep after each step
        self.timesteps.append(self.num_timesteps)  # Current timestep
        self.rewards.append(self.locals['rewards'][0])  # Reward for this timestep
        return True

# TensorBoard Callback (for logging)
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

# Create the environment
env = RummyEnv(max_rounds=100)
env = DummyVecEnv([lambda: env])

# Create the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_rummy_tensorboard/")

# Set up callbacks
reward_logger_callback = RewardLoggerCallback()
tensorboard_callback = TensorboardCallback()

# Train the model
model.learn(total_timesteps=50000, callback=[reward_logger_callback, tensorboard_callback])

# Visualize the training rewards over time
plt.plot(reward_logger_callback.timesteps, reward_logger_callback.rewards)
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Training Rewards Over Time')
plt.show()
