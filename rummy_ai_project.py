import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optuna

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

        if action < len(active_hand):  # Discard
            discarded_card = active_hand.pop(action)
            self.discard_pile.append(discarded_card)
            reward += 0.5
        elif action == len(active_hand):  # Draw from deck
            if len(self.deck) > 0:
                drawn_card = self.deck.pop()
                active_hand.append(drawn_card)
                reward += 0
            else:
                terminated = True
        else:
            reward -= 10  # Larger penalty for invalid actions

        reward += self._evaluate_hand(active_hand)

        if self._check_win(active_hand):
            terminated = True
            reward += 100

        self.current_turn = 3 - self.current_turn
        self.rounds += 1
        observation = self._get_observation()

        # Returning 5 values as required
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
        """
        Evaluate the hand and assign a reward based on advanced strategies:
        1. Prioritize forming pure sequences.
        2. Reward forming valid sets and impure sequences.
        3. Penalize holding high-point cards (10, J, Q, K, A).
        4. Reward strategic discarding of high-point cards not forming sequences.
        """
        reward = 0

        # Check for pure sequences
        if self._has_pure_sequence(hand):
            reward += 20  # Larger reward for forming pure sequences

        # Check for valid combinations (sets and impure sequences)
        if self._has_valid_combinations(hand):
            reward += 10

        # Penalize for holding high-point cards
        high_point_cards = [10, 11, 12, 13, 1]  # 10, J, Q, K, A
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward -= 2  # Penalize holding high-point cards

        # Encourage discarding high-point cards not in sequences
        if self.current_turn == 1:
            last_discard = self.discard_pile[-1] if self.discard_pile else None
            if last_discard and last_discard != 'joker' and last_discard[0] in high_point_cards:
                reward += 5  # Reward discarding high-point cards

        # Penalize for excessive cards (encourage efficient play)
        reward -= len(hand) * 0.2

        return reward


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

def optimize_hyperparameters(trial):
    print("Starting a new trial...")

    # Define the hyperparameters to optimize
    num_envs = trial.suggest_int("num_envs", 1, 2)  # Reduced for debugging
    batch_size = trial.suggest_categorical("batch_size", [128, 256])  # Reduced options for faster testing
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)

    log_dir = "./ppo_rummy_logs/optuna/"
    os.makedirs(log_dir, exist_ok=True)

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir,
                batch_size=batch_size, n_steps=1024, learning_rate=learning_rate, gamma=gamma)  # Reduced n_steps

    # Training the model with timeout limit
    try:
        model.learn(total_timesteps=100000)  # Cap total training time
    except Exception as e:
        print(f"Trial failed: {e}")
        env.close()
        return -float('inf')

    # Evaluate model performance
    total_rewards = []
    for _ in range(2):  # Evaluate across 2 episodes for faster feedback
        obs = env.reset()
        terminated = np.zeros(num_envs, dtype=bool)
        total_reward = np.zeros(num_envs)

        step_count = 0
        while not terminated.all() and step_count < 1000:  # Limit steps per evaluation
            action, _ = model.predict(obs)
            step_result = env.step(action)
            obs, reward, terminated, info = step_result[:4]
            total_reward += reward
            step_count += 1

        total_rewards.append(np.mean(total_reward))

    env.close()

    # Return the average reward as the objective value
    return np.mean(total_rewards)

# Modularized Training Function
def train_model(total_timesteps=5000, num_envs=2):
    if PPO is None or DummyVecEnv is None:
        print("Cannot execute training because required libraries are missing.")
        return

    log_dir = "./ppo_rummy_logs/"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir + "custom_metrics/")

    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
                batch_size=128, n_steps=1024, learning_rate=3e-4, gamma=0.99)  # Reduced batch size and steps

    live_plot = LivePlot()
    reward_logger = RewardLoggerCallback(writer, live_plot)

    model.learn(total_timesteps=total_timesteps, callback=reward_logger)

    # Save the model
    model.save("rummy_ai_model")

    # Close the writer
    writer.close()

    # Keep the plot open
    plt.ioff()
    plt.show()

def evaluate_model(model_path="rummy_ai_model", num_games=5):  # Reduced num_games for faster evaluation
    if PPO is None:
        print("Cannot evaluate because PPO is not available.")
        return

    model = PPO.load(model_path)
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])

    for game in range(num_games):
        obs = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated and step_count < 1000:  # Limit steps per game
            action, _ = model.predict(obs)
            step_result = env.step(action)

            obs, reward, terminated = step_result[:3]
            info = step_result[3] if len(step_result) > 3 else {}

            total_reward += reward
            step_count += 1

        print(f"Game {game + 1}/{num_games}: Total Reward = {total_reward}")

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_hyperparameters, n_trials=5, timeout=3600)  # Added timeout

    print("Best hyperparameters:", study.best_params)
    
    train_model()
    evaluate_model()