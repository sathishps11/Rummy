import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import optuna
from collections import Counter
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Placeholder for missing libraries
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    PPO = None
    DummyVecEnv = None
    SubprocVecEnv = None
    BaseCallback = None
    SummaryWriter = None
    print("The 'stable_baselines3' and 'torch' modules are required but not installed.")
    print("Please install them using the command: pip install stable-baselines3 torch")

# Rummy Environment Class
class RummyEnv(gym.Env):
    def __init__(self, max_rounds=100):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)
        self.action_space = spaces.Discrete(54)
        self.max_rounds = max_rounds
        self.reset()
        self.win_count = 0  # Track the number of wins
        self.loss_count = 0  # Track the number of losses

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.metrics = {
            "pure_sequences": 0,
            "valid_sets": 0,
            "strategic_discards": 0
        }
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
        obs = np.zeros(54, dtype=np.float32)  # Compact representation
        for card in self.hand_1:
            obs[self._card_to_index(card)] += 1
        return obs

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

        if self._check_win(active_hand):
            terminated = True
            reward += 100
            if self.current_turn == 1:
                self.win_count += 1
            else:
                self.loss_count += 1

        if action < len(active_hand):
            discarded_card = active_hand.pop(action)
            self.discard_pile.append(discarded_card)
            reward += 0.5
        elif action == len(active_hand):
            if len(self.deck) > 0:
                drawn_card = self.deck.pop()
                active_hand.append(drawn_card)
            else:
                terminated = True
        else:
            reward -= 10

        reward += self._evaluate_hand(active_hand)

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

    def _evaluate_hand(self, hand):
        reward = 0
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

        self.metrics["pure_sequences"] += 1 if has_pure_sequence else 0
        self.metrics["valid_sets"] += 1 if has_valid_combinations else 0

        if has_pure_sequence:
            reward += 50

        if has_valid_combinations:
            reward += 30

        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward -= 2

        if self.current_turn == 1:
            last_discard = self.discard_pile[-1] if self.discard_pile else None
            if last_discard and last_discard != 'joker' and last_discard[0] in high_point_cards:
                reward += 5

        reward -= len(hand) * 0.2

        return reward

    
def _is_in_combination(self, hand, card):
    """
    Check if the given card is part of a valid combination (set or sequence) in the hand.
    """
    rank_counts = {}
    for c in hand:
        if c != 'joker':
            rank = c[0]
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

    if card != 'joker':
        rank = card[0]
        if rank_counts.get(rank, 0) >= 3:
            return True

    sorted_hand = sorted([c for c in hand if c != 'joker'], key=lambda x: (x[1], x[0]))
    temp_seq = []
    for c in sorted_hand:
        if temp_seq and c[1] == temp_seq[-1][1] and c[0] == temp_seq[-1][0] + 1:
            temp_seq.append(c)
        else:
            if len(temp_seq) >= 3:
                return True
            temp_seq = [c]
    return len(temp_seq) >= 3



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

            if self.num_timesteps % 1000 == 0:
                avg_reward = np.mean(self.rewards[-100:])
                self.writer.add_scalar("Average Reward", avg_reward, self.num_timesteps)
                print(f"Step: {self.num_timesteps}, Avg Reward: {avg_reward:.2f}")
                self.live_plot.update(self.timesteps, self.rewards)

            return True

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


def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int('num_envs', 2, 4)
    batch_size = trial.suggest_int('batch_size', 64, 512)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.99)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['constant', 'exponential_decay'])

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, gamma=gamma,
                batch_size=batch_size, n_steps=1024)

    if lr_scheduler == 'exponential_decay':
        model.learning_rate = learning_rate * (0.95 ** np.arange(1000))

    model.learn(total_timesteps=5000)

    total_rewards = []
    for _ in range(5):
        obs = env.reset()
        terminated = np.zeros(num_envs, dtype=bool)
        total_reward = np.zeros(num_envs)
        while not terminated.all():
            action, _ = model.predict(obs)
            obs, reward, terminated, info = env.step(action)
            total_reward += reward
        total_rewards.append(np.mean(total_reward))

    env.close()
    return np.mean(total_rewards)



# Reward Function Improvement with dynamic hand evaluation
def custom_evaluate_hand(self, hand):
    reward = 0
    has_pure_sequence = self._has_pure_sequence(hand)
    has_valid_combinations = self._has_valid_combinations(hand)

    if has_pure_sequence:
        reward += 50
    if has_valid_combinations:
        reward += 30

    high_point_cards = [10, 11, 12, 13, 1]
    for card in hand:
        if card != 'joker' and card[0] in high_point_cards:
            reward -= 2

    last_discard = self.discard_pile[-1] if self.discard_pile else None
    if last_discard and last_discard[0] in high_point_cards:
        reward += 5

    reward -= len(hand) * 0.2

    return reward

RummyEnv._evaluate_hand = custom_evaluate_hand


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, obs):
        actions = [model.predict(obs)[0] for model in self.models]
        return Counter(actions).most_common(1)[0][0]


def transfer_learning(model, env, num_steps=10000):
    pretrained_model = PPO.load('pretrained_rummy_model')
    model.set_parameters(pretrained_model.get_parameters())
    model.learn(total_timesteps=num_steps)



# Training function with multiple algorithms and optimizations
def train_model():
    log_dir = "./ppo_rummy_logs/"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir + "custom_metrics/")

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(2)])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

    model.learn(total_timesteps=100000)

    model.save("rummy_ai_model")

    transfer_learning(model, env)

    writer.close()
    
def optimize_rewards(trial):
    w_pure_seq = trial.suggest_int("w_pure_seq", 20, 100)
    w_valid_set = trial.suggest_int("w_valid_set", 10, 80)
    w_high_point_penalty = trial.suggest_int("w_high_point_penalty", -5, -1)
    w_discard_high = trial.suggest_int("w_discard_high", 1, 10)
    w_stall_penalty = trial.suggest_int("w_stall_penalty", -20, -5)

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(2)])

    model = PPO("MlpPolicy", env, verbose=0, batch_size=128, gamma=0.99)
    model.learn(total_timesteps=5000)

    total_rewards = []
    for _ in range(5):
        obs = env.reset()
        terminated = np.zeros(2, dtype=bool)
        total_reward = np.zeros(2)
        while not terminated.all():
            action, _ = model.predict(obs)
            obs, reward, terminated, info = env.step(action)
            total_reward += reward
        total_rewards.append(np.mean(total_reward))

    return np.mean(total_rewards)


# Optimize using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(optimize_hyperparameters, n_trials=100)
print("Best parameters:", study.best_params)


# Train the model after hyperparameter optimization
train_model()



def optimize_rewards(trial):
    w_pure_seq = trial.suggest_int("w_pure_seq", 20, 100)
    w_valid_set = trial.suggest_int("w_valid_set", 10, 80)
    w_high_point_penalty = trial.suggest_int("w_high_point_penalty", -5, -1)
    w_discard_high = trial.suggest_int("w_discard_high", 1, 10)
    w_stall_penalty = trial.suggest_int("w_stall_penalty", -20, -5)
    w_joker_usage = trial.suggest_int("w_joker_usage", 5, 20)

    def custom_evaluate_hand(self, hand):
        reward = 0
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

        if has_pure_sequence:
            reward += w_pure_seq
        if has_valid_combinations:
            reward += w_valid_set

        # Penalize holding high-point cards
        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward += w_high_point_penalty

        # Reward discarding high-point cards
        last_discard = self.discard_pile[-1] if self.discard_pile else None
        if last_discard and last_discard[0] in high_point_cards:
            reward += w_discard_high

        # Penalize stalling
        if not has_pure_sequence and not has_valid_combinations:
            reward += w_stall_penalty

        # Reward for using jokers in valid combinations
        jokers_used = sum(1 for card in hand if card == 'joker')
        reward += w_joker_usage * jokers_used

        return reward

    RummyEnv._evaluate_hand = custom_evaluate_hand
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)

    win_rate = evaluate_model(model, num_games=50)
    env.close()
    return win_rate


def evaluate_model(model, num_games=50):
    """
    Evaluate the model's performance by simulating a number of games.
    Returns the win rate as the evaluation metric.
    """
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])
    wins = 0

    for game in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        if rewards[0] > 0:  # Assuming positive rewards indicate a win
            wins += 1

    env.close()
    return wins / num_games  # Win rate


if __name__ == '__main__':
    # No need for freeze_support() unless creating an executable
    # freeze_support()  # Remove this line

    def optimize_hyperparameters(trial):
        num_envs = trial.suggest_int('num_envs', 2, 4)
        batch_size = trial.suggest_int('batch_size', 64, 512)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        gamma = trial.suggest_uniform('gamma', 0.8, 0.99)
        lr_scheduler = trial.suggest_categorical('lr_scheduler', ['constant', 'exponential_decay'])

        # Create the environment inside the main block to avoid process spawn issues
        env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])

        # Code to set up and train the model with the selected hyperparameters
        model = PPO("MlpPolicy", env, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate, verbose=0)
        model.learn(total_timesteps=5000)

        win_rate = evaluate_model(model)
        return win_rate

    # Create and optimize the study with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_hyperparameters, n_trials=100)

    # Optionally print the best parameters found by Optuna
    print("Best parameters:", study.best_params)

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

# Modularized Training Function
def train_model(total_timesteps=5000, num_envs=2):
    log_dir = "./ppo_rummy_logs/"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir + "custom_metrics/")

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

    live_plot = LivePlot()
    reward_logger = RewardLoggerCallback(writer, live_plot)

    model.learn(total_timesteps=total_timesteps, callback=reward_logger)

    # Save the model
    model.save("rummy_ai_model")

    writer.close()

    # Keep the plot open
    plt.ioff()
    plt.show()

