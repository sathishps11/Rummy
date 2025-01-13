import os
import numpy as np
import random
from collections import Counter
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import optuna
from pettingzoo.utils.conversions import parallel_wrapper_fn

# Define your custom environment
class MultiAgentRummyEnv(ParallelEnv):
    metadata = {'render_modes': ['human']}

    def __init__(self, max_rounds=100, num_players=2):
        super().__init__()

        # Game settings
        self.max_rounds = max_rounds
        self.num_players = num_players

        # Define action and observation spaces
        self.action_space = spaces.Discrete(54)  # Example action space size: 54 possible actions (cards)
        self.observation_space = spaces.Box(low=0, high=1, shape=(54 + 2 + 54 + 4,), dtype=np.float32)  # 54 cards, 2 ratios, 3 recent discard cards, 4 additional features

        # Initialize game state
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.discard_pile = []
        self.hands = {agent: [] for agent in self.agents}
        self.deck = []
        self.rounds = 0

        # Metrics for tracking gameplay performance
        self.metrics = {
            "pure_sequences": 0,
            "valid_sets": 0,
            "blocks": 0,
            "high_point_penalty": 0,
            "help_opponent": 0
        }

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.rounds = 0
        self.deck = self._create_deck()
        random.shuffle(self.deck)

        self.hands = {f"player_{i}": [self.deck.pop() for _ in range(13)] for i in range(self.num_players)}
        self.discard_pile = [self.deck.pop()]

        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.hands.keys()}
        self.dones = {agent: False for agent in self.hands.keys()}

        return self.observe_all()

    def observe_all(self):
        """
        Generate observations for all agents.
        """
        return {agent: self.observe(agent) for agent in self.hands.keys()}

    def observe(self, agent):
        """
        Return the observation for a specific agent.
        """
        hand = self.hands[agent]
        obs = np.zeros(54, dtype=np.float32)

        # Encode the player's hand into the observation vector
        for card in hand:
            obs[self._card_to_index(card)] += 1

        # Add metadata: Deck ratio, round ratio, discard pile history
        deck_ratio = len(self.deck) / 54
        round_ratio = self.rounds / self.max_rounds

        discard_obs = np.zeros(54, dtype=np.float32)
        for card in self.discard_pile[-3:]:
            discard_obs[self._card_to_index(card)] += 1

        # Add additional features (number of players, deck size, etc.)
        additional_features = np.zeros(4, dtype=np.float32)
        additional_features[0] = len(self.hands)  # Number of players
        additional_features[1] = len(self.deck)   # Number of remaining cards in the deck
        additional_features[2] = self.rounds / self.max_rounds  # Current round ratio
        additional_features[3] = self.rounds  # Current round number

        # Concatenate all features into the observation
        obs = np.concatenate([obs, [deck_ratio, round_ratio], discard_obs, additional_features]).astype(np.float32)

        return obs


    def step(self, actions):
        """
        Perform actions for all agents and update the environment state.
        """
        rewards = {}
        dones = {}

        for agent, action in actions.items():
            if self.dones[agent]:
                continue

            hand = self.hands[agent]

            # Process action: Discard or draw
            if action < len(hand):
                discarded_card = hand.pop(action)
                self.discard_pile.append(discarded_card)
                rewards[agent] = self._evaluate_hand(hand)
            elif action == len(hand):  # Draw action
                if self.deck:
                    hand.append(self.deck.pop())

            self.dones[agent] = self.rounds >= self.max_rounds or self._check_win(hand)

        # Update rounds and determine if all players are done
        self.rounds += 1
        all_done = all(self.dones.values())

        for agent in self.hands.keys():
            dones[agent] = all_done

        return self.observe_all(), rewards, dones, {}

    def _circular_next_agent(self):
        """
        Move to the next agent in a circular manner.
        """
        next_index = (self.agents.index(self.agent_selection) + 1) % len(self.agents)
        self.agent_selection = self.agents[next_index]

    def _create_deck(self):
        """
        Create a standard deck of Rummy cards, including jokers.
        """
        suits = ['♥', '♦', '♣', '♠']
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2

    def _card_to_index(self, card):
        """
        Convert a card to an index for observation encoding.
        """
        if card == 'joker':
            return 52
        rank, suit = card
        suit_offset = {'♥': 0, '♦': 13, '♣': 26, '♠': 39}[suit]
        return suit_offset + rank - 1

    def _check_win(self, hand):
        """
        Check if the hand meets the winning conditions.
        """
        return self._has_pure_sequence(hand) and self._has_valid_combinations(hand)

    def _has_pure_sequence(self, hand):
        """
        Check for a pure sequence in the hand.
        """
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
        """
        Check for valid sets or combinations in the hand.
        """
        rank_counts = Counter(card[0] for card in hand if card != 'joker')
        return sum(1 for count in rank_counts.values() if count >= 3) >= 2

    def _evaluate_hand(self, hand):
        """
        Evaluate the current player's hand and assign rewards.
        """
        reward = 0
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

        # Update metrics
        self.metrics["pure_sequences"] += 1 if has_pure_sequence else 0
        self.metrics["valid_sets"] += 1 if has_valid_combinations else 0

        # Reward valid sequences and combinations
        if has_pure_sequence:
            reward += 50
        if has_valid_combinations:
            reward += 30

        # Penalize holding high-point cards
        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward -= 2

        # Reward for discarding high-point cards
        last_discard = self.discard_pile[-1] if self.discard_pile else None
        if last_discard and last_discard[0] in high_point_cards:
            reward += 5

        # Reward for using jokers strategically
        jokers_used = sum(1 for card in hand if card == 'joker')
        reward += 10 * jokers_used

        # Penalize holding too many cards late in the game
        if self.rounds > self.max_rounds * 0.75:
            reward -= len(hand) * 0.5

        return reward
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import optuna

# Define RewardLoggerCallback
class RewardLoggerCallback(BaseCallback):
    def __init__(self, writer, plot_dashboard=None):
        super(RewardLoggerCallback, self).__init__()
        self.writer = writer
        self.plot_dashboard = plot_dashboard

    def _on_step(self):
        # Log reward to TensorBoard
        if self.locals.get('rewards') is not None:
            for i, reward in enumerate(self.locals['rewards']):
                self.writer.add_scalar(f"agent_{i}/reward", reward, self.num_timesteps)
        return True

# Function to wrap the environment (assuming you already have this function defined)
def parallel_wrapper_fn(env_fn):
    def wrapper():
        return env_fn()
    return wrapper

# Training the model
def train_multiagent_model(env_fn, total_timesteps=5000, num_envs=2):
    log_dir = "./ppo_rummy_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir + "custom_metrics/")

    try:
        # Create parallel environments
        env = SubprocVecEnv([lambda: parallel_wrapper_fn(env_fn)() for _ in range(num_envs)])

        # Initialize the PPO model
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

        # Initialize the reward logger callback
        reward_logger = RewardLoggerCallback(writer=writer, plot_dashboard=None)

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=reward_logger)

        # Save the trained model
        model.save("rummy_ai_model")

        # Close the writer
        writer.close()

        # Close the plot
        plt.ioff()
        plt.show()
    
    except Exception as e:
        print(f"Error during training: {e}")

# Example usage of training function
train_multiagent_model(MultiAgentRummyEnv)

# Example of hyperparameter optimization function using optuna
def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int("num_envs", 2, 4)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)

    try:
        # Create the environment
        env = [parallel_wrapper_fn(MultiAgentRummyEnv)(max_rounds=100, num_players=2) for _ in range(num_envs)]

        # Placeholder for custom RL model initialization and training using the PettingZoo environment
        # Your model would be trained here. This might be custom PPO, DQN, or any other RL model
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma)

        # Train the model
        model.learn(total_timesteps=5000)

        # Evaluate the model after training
        win_rate = evaluate_model(model)
        return win_rate
    except Exception as e:
        raise RuntimeError("Error during hyperparameter optimization.") from e


# Custom evaluation function to test the trained model's performance
def evaluate_model(model, num_games=50):
    """
    Evaluate the model's performance by simulating a number of games.
    Returns the win rate as the evaluation metric.
    """
    try:
        env = [parallel_wrapper_fn(MultiAgentRummyEnv)(max_rounds=100, num_players=2)]

        wins = 0
        for game in range(num_games):
            obs = env.reset()
            dones = [False]
            while not all(dones):
                actions, _ = model.predict(obs)  # This assumes the model has a predict method
                obs, rewards, dones, infos = env.step(actions)
                if any(reward > 0 for reward in rewards):
                    wins += 1

        env.close()
        return wins / num_games  # Win rate
    except Exception as e:
        raise RuntimeError("Error during model evaluation.") from e


import numpy as np
import matplotlib.pyplot as plt
import optuna
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo import envs
from pettingzoo.utils import agent_selector
from pettingzoo.utils import random_seeds
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Custom Reward Evaluation Logic
def custom_evaluate_hand(self, hand, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage):
    reward = 0
    has_pure_sequence = self._has_pure_sequence(hand)
    has_valid_combinations = self._has_valid_combinations(hand)

    # Reward pure sequences and valid sets
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

    # Penalize stalling if no pure sequences or valid sets
    if not has_pure_sequence and not has_valid_combinations:
        reward += w_stall_penalty

    # Reward for using jokers in valid combinations
    jokers_used = sum(1 for card in hand if card == 'joker')
    reward += w_joker_usage * jokers_used

    return reward

# Updating the environment's reward function dynamically
def update_reward_function(env, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage):
    env._evaluate_hand = lambda hand: custom_evaluate_hand(hand, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage)

# Define Optimization Function
def optimize_rewards(trial):
    # Reward hyperparameters
    w_pure_seq = trial.suggest_int("w_pure_seq", 25, 35)
    w_valid_set = trial.suggest_int("w_valid_set", 45, 55)
    w_high_point_penalty = trial.suggest_int("w_high_point_penalty", -6, -4)
    w_discard_high = trial.suggest_int("w_discard_high", 7, 11)
    w_stall_penalty = trial.suggest_int("w_stall_penalty", -7, -4)
    w_joker_usage = trial.suggest_int("w_joker_usage", 10, 15)

    # Create Parallel Environments with PettingZoo
    num_envs = trial.suggest_int("num_envs", 2, 4)
    env = SubprocVecEnv([lambda: MultiAgentRummyEnv(max_rounds=100) for _ in range(num_envs)])

    # Update the reward function with the current hyperparameters
    update_reward_function(env, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage)

    # Model configuration
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    
    # Use PettingZoo-compatible agent configuration
    agent = PPO("MlpPolicy", env, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)

    # Train the model
    agent.learn(total_timesteps=5000)

    # Evaluate the model using win rate as the metric
    win_rate = evaluate_model(agent, num_games=50)
    env.close()
    return win_rate

# Example usage of hyperparameter optimization
# Uncomment to run optimization
# study = optuna.create_study(direction="maximize")
# study.optimize(optimize_rewards, n_trials=10)

from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo import PPO
from pettingzoo.utils.env import DummyVecEnv, SubprocVecEnv
import optuna
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from MultiAgentRummy import MultiAgentRummyEnv  # Import your custom environment

# Class for live plotting during training
class LivePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.reward_history = []
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Live Training Reward Plot")
        plt.ion()

    def update(self, reward):
        try:
            reward = np.array(reward).flatten()
            if reward.ndim != 1:
                print(f"Error: Expected 1D array, but got {reward.ndim}D.")
                return

            if reward.size == 1:
                self.reward_history.append(reward[0])
            else:
                self.reward_history.extend(reward)

            self.ax.clear()
            self.ax.plot(self.reward_history)
            self.ax.set_xlabel("Training Steps")
            self.ax.set_ylabel("Average Reward")
            self.ax.set_title("Live Training Progress")
            plt.draw()
            plt.pause(0.01)
        except Exception as e:
            print(f"Error during update: {e}")

    def show_plot(self):
        plt.show()

# Class to handle a more comprehensive live dashboard
class Dashboard:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.ion()
        self.reward_ax = self.axes[0, 0]
        self.reward_ax.set_title("Reward Over Timesteps")
        self.reward_ax.set_xlabel("Timesteps")
        self.reward_ax.set_ylabel("Rewards")
        self.reward_line, = self.reward_ax.plot([], [], lw=2, label="Rewards")
        self.reward_avg_line, = self.reward_ax.plot([], [], lw=2, label="Average Reward")
        self.reward_ax.legend()

        self.win_ax = self.axes[0, 1]
        self.win_ax.set_title("Win/Loss Trend")
        self.win_ax.set_xlabel("Episodes")
        self.win_ax.set_ylabel("Cumulative Count")
        self.win_line, = self.win_ax.plot([], [], lw=2, label="Wins")
        self.loss_line, = self.win_ax.plot([], [], lw=2, label="Losses")
        self.win_ax.legend()

        self.action_ax = self.axes[1, 0]
        self.action_ax.set_title("Action Selection Frequency")
        self.action_ax.set_xlabel("Actions")
        self.action_ax.set_ylabel("Frequency")
        self.action_bars = None

        self.metric_ax = self.axes[1, 1]
        self.metric_ax.set_title("Key Metric Trends")
        self.metric_ax.set_xlabel("Timesteps")
        self.metric_ax.set_ylabel("Counts")
        self.pure_seq_line, = self.metric_ax.plot([], [], lw=2, label="Pure Sequences")
        self.valid_sets_line, = self.metric_ax.plot([], [], lw=2, label="Valid Sets")
        self.metric_ax.legend()

        self.timesteps = []
        self.rewards = []
        self.avg_rewards = []
        self.wins = []
        self.losses = []
        self.pure_sequences = []
        self.valid_sets = []
        self.action_counts = Counter()

    def update(self, timestep, reward, avg_reward, win_count, loss_count, metrics, action):
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.wins.append(win_count)
        self.losses.append(loss_count)
        self.pure_sequences.append(metrics["pure_sequences"])
        self.valid_sets.append(metrics["valid_sets"])
        self.action_counts[action] += 1

        self.reward_line.set_data(self.timesteps, self.rewards)
        self.reward_avg_line.set_data(self.timesteps, self.avg_rewards)
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()

        self.win_line.set_data(range(len(self.wins)), self.wins)
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.win_ax.relim()
        self.win_ax.autoscale_view()

        if self.action_bars:
            for bar, count in zip(self.action_bars, self.action_counts.values()):
                bar.set_height(count)
        else:
            self.action_bars = self.action_ax.bar(self.action_counts.keys(), self.action_counts.values())
        self.action_ax.relim()
        self.action_ax.autoscale_view()

        self.pure_seq_line.set_data(self.timesteps, self.pure_sequences)
        self.valid_sets_line.set_data(self.timesteps, self.valid_sets)
        self.metric_ax.relim()
        self.metric_ax.autoscale_view()

        plt.pause(0.01)
        plt.draw()

import numpy as np
import matplotlib.pyplot as plt
import optuna
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo import envs
from pettingzoo.utils import agent_selector
from pettingzoo.utils import random_seeds
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

# Custom reward evaluation logic
def custom_evaluate_hand(self, hand, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage):
    reward = 0
    has_pure_sequence = self._has_pure_sequence(hand)
    has_valid_combinations = self._has_valid_combinations(hand)

    # Reward pure sequences and valid sets
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

    # Penalize stalling if no pure sequences or valid sets
    if not has_pure_sequence and not has_valid_combinations:
        reward += w_stall_penalty

    # Reward for using jokers in valid combinations
    jokers_used = sum(1 for card in hand if card == 'joker')
    reward += w_joker_usage * jokers_used

    return reward

# Update environment with new reward function
def update_reward_function(env, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage):
    env._evaluate_hand = lambda hand: custom_evaluate_hand(hand, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage)

# Define environment creation function to avoid lambda inside SubprocVecEnv
def create_env_fn():
    return MultiAgentRummyEnv(max_rounds=100)

# Define optimization function for the rewards
def optimize_rewards(trial):
    # Reward hyperparameters
    w_pure_seq = trial.suggest_int("w_pure_seq", 25, 35)
    w_valid_set = trial.suggest_int("w_valid_set", 45, 55)
    w_high_point_penalty = trial.suggest_int("w_high_point_penalty", -6, -4)
    w_discard_high = trial.suggest_int("w_discard_high", 7, 11)
    w_stall_penalty = trial.suggest_int("w_stall_penalty", -7, -4)
    w_joker_usage = trial.suggest_int("w_joker_usage", 10, 15)

    # Create Parallel Environments with PettingZoo
    num_envs = trial.suggest_int("num_envs", 2, 4)
    
    # Use list comprehension to create multiple environments
    env = SubprocVecEnv([create_env_fn for _ in range(num_envs)])

    # Update the reward function with the current hyperparameters
    update_reward_function(env, w_pure_seq, w_valid_set, w_high_point_penalty, w_discard_high, w_stall_penalty, w_joker_usage)

    # Model configuration
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    
    # Create and train the model
    model = PPO("MlpPolicy", env, verbose=0, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)
    model.learn(total_timesteps=5000)

    # Evaluate the model using win rate as the metric
    win_rate = evaluate_model(model, num_games=50)
    env.close()
    return win_rate

import multiprocessing

if __name__ == "__main__":
    # Only include freeze_support if you are packaging the script into an executable
    if multiprocessing.get_start_method() == 'spawn':
        multiprocessing.freeze_support()
    
    # Training and optimization code
    train_multiagent_model(MultiAgentRummyEnv)

    # Hyperparameter optimization using optuna
    study = optuna.create_study(direction="maximize", study_name="Rummy Reward Optimization")
    study.optimize(optimize_rewards, n_trials=2)

    best_params = study.best_params
    env = SubprocVecEnv([create_env_fn for _ in range(best_params.get("num_envs", 1))])

    # Final model training with optimized parameters
    model = PPO("MlpPolicy", env, verbose=1, batch_size=128, gamma=0.99)
    model.learn(total_timesteps=50000)

    # Save the model
    model.save("rummy_ai_model")

    # Plot results (example, you can update this based on your needs)
    plt.show()
