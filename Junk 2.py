import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


class RummyEnv(gym.Env):
    def __init__(self, max_rounds=100):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(54,), dtype=np.float32)
        self.action_space = spaces.Discrete(54)
        self.max_rounds = max_rounds
        self.reset()
        self.win_count = 0
        self.loss_count = 0

    def reset(self, seed=None, options=None):
        # Initializes game state, shuffles deck, and deals cards
        self.metrics = {"pure_sequences": 0, "valid_sets": 0, "strategic_discards": 0}
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.hand_1 = [self.deck.pop() for _ in range(13)]
        self.hand_2 = [self.deck.pop() for _ in range(13)]
        self.discard_pile = [self.deck.pop()]
        self.current_turn = 1
        self.rounds = 0
        return self._get_observation(), {}

    def _create_deck(self):
        # Creates a deck of 52 cards and 2 jokers
        suits = ['♥', '♦', '♣', '♠']
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2

    def _get_observation(self):
        # Returns a compact representation of the current hand of player 1
        obs = np.zeros(54, dtype=np.float32)
        for card in self.hand_1:
            obs[self._card_to_index(card)] += 1
        return obs

    def _card_to_index(self, card):
        # Converts a card to an index
        if card == 'joker':
            return 52
        rank, suit = card
        suit_offset = {'♥': 0, '♦': 13, '♣': 26, '♠': 39}[suit]
        return suit_offset + rank - 1

    def step(self, action):
        # Executes one action (discard or draw card)
        reward = 0
        terminated = False
        truncated = False

        # Select active hand based on whose turn it is
        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        # Check for win condition
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
            # Draw card if no discard is made
            if len(self.deck) > 0:
                drawn_card = self.deck.pop()
                active_hand.append(drawn_card)
            else:
                terminated = True
        else:
            reward -= 10  # Invalid action

        reward += self._evaluate_hand(active_hand)

        if self._check_win(active_hand):
            terminated = True
            reward += 100

        # Switch turns and increment rounds
        self.current_turn = 3 - self.current_turn
        self.rounds += 1

        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def _check_win(self, hand):
        # Check if a hand satisfies the win condition (pure sequence + valid sets)
        return self._has_pure_sequence(hand) and self._has_valid_combinations(hand)

    def _has_pure_sequence(self, hand):
        # Checks if there is a pure sequence in the hand
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
        # Checks if there are at least two valid sets in the hand
        rank_counts = {}
        for card in hand:
            if card != 'joker':
                rank = card[0]
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
        return sum(1 for count in rank_counts.values() if count >= 3) >= 2

    def _evaluate_hand(self, hand):
        # Custom reward function based on hand composition
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


class RewardLoggerCallback(BaseCallback):
    def __init__(self, writer, live_plot, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.timesteps = []
        self.rewards = []
        self.live_plot = live_plot

    def _on_step(self) -> bool:
        # Log the reward and update the plot at each step
        current_timestep = self.num_timesteps
        reward = self.locals.get("rewards", [0])[0]

        self.timesteps.append(current_timestep)
        self.rewards.append(reward)

        if current_timestep % 1000 == 0:
            avg_reward = np.mean(self.rewards[-100:])
            self.writer.add_scalar("Average Reward", avg_reward, current_timestep)
            self.live_plot.update(self.timesteps, self.rewards)

        return True


class LivePlot:
    def __init__(self):
        # Initializes the live plot with default configurations
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
        # Updates the live plot with new data points
        if len(timesteps) > 0 and len(rewards) > 0:
            self.line.set_data(timesteps, rewards)
            self.ax.set_xlim(0, max(timesteps) + 100)  # Adjust X-axis dynamically
            self.ax.set_ylim(min(rewards) - 10, max(rewards) + 10)  # Adjust Y-axis dynamically
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


def evaluate_model(model, num_games=50):
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])
    wins = 0

    for game in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        if rewards[0] > 0:
            wins += 1

    env.close()
    return wins / num_games


def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int('num_envs', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['constant', 'exponential_decay'])

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])

    model = PPO("MlpPolicy", env, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate, verbose=0)

    if lr_scheduler == 'exponential_decay':
        model.learning_rate = learning_rate * (0.95 ** np.arange(1000))

    model.learn(total_timesteps=5000)

    win_rate = evaluate_model(model)
    env.close()
    return win_rate


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

        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward += w_high_point_penalty

        last_discard = self.discard_pile[-1] if self.discard_pile else None
        if last_discard and last_discard[0] in high_point_cards:
            reward += w_discard_high

        if not has_pure_sequence and not has_valid_combinations:
            reward += w_stall_penalty

        jokers_used = sum(1 for card in hand if card == 'joker')
        reward += w_joker_usage * jokers_used

        return reward

    RummyEnv._evaluate_hand = custom_evaluate_hand

    num_envs = trial.suggest_int("num_envs", 2, 4)
    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    model = PPO("MlpPolicy", env, verbose=0, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)

    model.learn(total_timesteps=5000)

    win_rate = evaluate_model(model, num_games=50)
    env.close()
    return win_rate


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="Rummy Reward Optimization")
    study.optimize(optimize_rewards, n_trials=1)

    print("Best parameters:", study.best_params)

    import optuna.visualization as vis
    vis.plot_param_importances(study).show()
    vis.plot_optimization_history(study).show()

    def train_model(total_timesteps=5000, num_envs=2):
        log_dir = "./ppo_rummy_logs/"
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir + "custom_metrics/")

        env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

        live_plot = LivePlot()
        reward_logger = RewardLoggerCallback(writer, live_plot)

        model.learn(total_timesteps=total_timesteps, callback=reward_logger)

        model.save("rummy_ai_model")
        writer.close()

        plt.ioff()
        plt.show()

    train_model(total_timesteps=5000, num_envs=study.best_params['num_envs'])



