import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RummyEnv(gym.Env):
    def __init__(self):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)
        self.action_space = spaces.Discrete(108)
        self.rng = np.random.default_rng()
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

        observation = self._get_observation()
        return observation, {}

    def _create_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
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
        suit_offset = {'hearts': 0, 'diamonds': 13, 'clubs': 26, 'spades': 39}[suit]
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

        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        if action_type == 0:
            if 0 <= card_index < len(active_hand):
                discarded_card = active_hand.pop(card_index)
                self.discard_pile.append(discarded_card)
                reward += 1
            else:
                reward -= 1
        elif action_type == 1:
            if len(self.draw_pile) > 0:
                active_hand.append(self.draw_pile.pop())
            else:
                terminated = True

        if len(self.hand_1) == 0 or len(self.hand_2) == 0:
            terminated = True
            reward += 20

        reward += self._evaluate_hand(active_hand)

        self.current_turn = 3 - self.current_turn

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _evaluate_hand(self, hand):
        return 10 * self._count_pure_sequences(hand) + 5 * self._count_impure_sequences(hand)

    def _count_pure_sequences(self, hand):
        return len(hand) // 5

    def _count_impure_sequences(self, hand):
        return len(hand) // 10


if __name__ == "__main__":
    env = DummyVecEnv([lambda: RummyEnv()])
    model = PPO('MlpPolicy', env, verbose=1)

    print("Training AI vs AI...")

    model.learn(total_timesteps=100000)
    model.save("rummy_ai_model")

    num_matches = 100

    for match in range(num_matches):
        obs = env.reset()  # Only unpack the observation
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated = env.step(action)  # Correct unpacking for 4 values
            done = terminated or truncated
            print(f"Match {match + 1}, Action: {action}, Reward: {reward[0]}, Done: {done}")
