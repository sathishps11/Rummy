import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class RummyEnv(gym.Env):
    def __init__(self):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)
        self.action_space = spaces.Discrete(54)
        self.rng = np.random.default_rng()  # Random number generator
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.draw_pile = self._create_deck()
        self.rng.shuffle(self.draw_pile)
        self.hand = self._draw_initial_hand()
        self.discard_pile = []
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
        obs = [0] * 54
        for card in self.hand:
            obs[self._card_to_index(card)] = 1
        return np.array(obs, dtype=np.float32)

    def _card_to_index(self, card):
        if card == 'joker':
            return 52
        rank, suit = card
        suit_offset = {'hearts': 0, 'diamonds': 13, 'clubs': 26, 'spades': 39}[suit]
        return suit_offset + rank - 1

    def _evaluate_hand(self):
        return len(self.hand) // 3

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False  # Always False unless implementing truncation logic

        if 0 <= action < len(self.hand):
            discarded_card = self.hand.pop(action)
            self.discard_pile.append(discarded_card)
        else:
            reward -= 1

        if len(self.draw_pile) > 0:
            self.hand.append(self.draw_pile.pop())
        else:
            terminated = True

        reward += self._evaluate_hand()

        done = terminated
        observation = self._get_observation()
        return observation, reward, done, truncated, {}

# Main script
if __name__ == "__main__":
    env = DummyVecEnv([lambda: RummyEnv()])
    model = PPO('MlpPolicy', env, verbose=1)

    print("Training AI...")
    model.learn(total_timesteps=100000)
    model.save("rummy_ai_model")

    result = env.reset()
    if isinstance(result, tuple):
        obs, _ = result
    else:
        obs = result

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)

       # Ensure action is a scalar, no need for the check
        action = int(action)  # Cast action to integer if it’s not already


        action = np.array([action])  # Wrap action in a 1D array
        obs, reward, done, info = env.step(action)

        obs = obs[0]
        reward = reward[0]
        done = done[0]

        print(f"Action: {action[0]}, Reward: {reward}, Done: {done}, Info: {info}")
