import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Placeholder for missing libraries
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    PPO = None
    DummyVecEnv = None
    BaseCallback = None
    SummaryWriter = None
    print("The 'stable_baselines3' and 'torch' modules are required but not installed.")
    print("Please install them using the command: pip install stable-baselines3 torch")

# Rummy Environment Class
class RummyEnv(gym.Env):
    def __init__(self, max_rounds=100):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)
        self.action_space = spaces.Discrete(108)
        self.max_rounds = max_rounds
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.hand_1 = [self.deck.pop() for _ in range(13)]
        self.hand_2 = [self.deck.pop() for _ in range(13)]
        self.discard_pile = [self.deck.pop()]
        self.current_turn = 1
        self.rounds = 0
        return self._get_observation(), {}

    def _create_deck(self):
        suits = ['♥', '♦', '♣', '♠']
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2

    def _get_observation(self):
        obs = [0] * 108
        for card in self.hand_1:
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
        reward = 0
        terminated = False
        truncated = False

        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        if action < 54:  # Discard
            if 0 <= action < len(active_hand):
                discarded_card = active_hand.pop(action)
                self.discard_pile.append(discarded_card)
                reward += 1
            else:
                reward -= 5
        else:  # Draw
            if len(self.deck) > 0:
                drawn_card = self.deck.pop()
                active_hand.append(drawn_card)
                reward += 0
            else:
                terminated = True

        if self._check_win(active_hand):
            terminated = True
            reward += 100

        self.current_turn = 3 - self.current_turn
        self.rounds += 1
        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def _check_win(self, hand):
        return self._has_pure_sequence(hand) and self._has_valid_combinations(hand)

    def _has_pure_sequence(self, hand):
        sorted_hand = sorted([card for card in hand if card != 'joker'], key=lambda x: (x[1], x[0]))
        temp_seq = []
        for card in sorted_hand:
            if temp_seq and card[1] == temp_seq[-1][1] and card[0] == temp_seq[-1][0] + 1:
                temp_seq.append(card)
            else:
                if len(temp_seq) >= 3:
                    return True
                temp_seq = [card]
        return len(temp_seq) >= 3

    def _has_valid_combinations(self, hand):
        rank_counts = {}
        for card in hand:
            if card != 'joker':
                rank = card[0]
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
        return sum(1 for count in rank_counts.values() if count >= 3) >= 2

# Callbacks for Logging
if BaseCallback is not None:
    class RewardLoggerCallback(BaseCallback):
        def __init__(self, writer, live_plot, verbose=0):
            super().__init__(verbose)
            self.writer = writer
            self.timesteps = []
            self.rewards = []
            self.live_plot = live_plot

        def _on_step(self) -> bool:
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(self.locals['rewards'][0])

            # Log average reward every 1000 steps
            if self.num_timesteps % 1000 == 0:
                avg_reward = np.mean(self.rewards[-100:])  # Average over the last 100 steps
                self.writer.add_scalar("Average Reward", avg_reward, self.num_timesteps)
                print(f"Step: {self.num_timesteps}, Avg Reward: {avg_reward:.2f}")
                self.live_plot.update(self.timesteps, self.rewards)

            return True

# Live Plot Class
class LivePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Rewards")
        self.ax.set_title("Live Training Progress")
        plt.ion()
        plt.show()

    def update(self, timesteps, rewards):
        self.line.set_data(timesteps, rewards)
        self.ax.set_xlim(0, max(timesteps) + 1000)
        self.ax.set_ylim(0, max(rewards) + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Training Workflow
if __name__ == "__main__":
    if PPO is None or DummyVecEnv is None:
        print("Cannot execute training because required libraries are missing.")
    else:
        log_dir = "./ppo_rummy_logs/"
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir + "custom_metrics/")

        env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

        live_plot = LivePlot()
        reward_logger = RewardLoggerCallback(writer, live_plot)

        model.learn(total_timesteps=50000, callback=reward_logger)

        # Save the model
        model.save("rummy_ai_model")

        # Close the writer
        writer.close()

        # Keep the plot open
        plt.ioff()
        plt.show()
