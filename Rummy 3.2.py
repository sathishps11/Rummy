import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optuna
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from rummy_env import RummyEnv

# Import the RummyEnv from the correct module
from rummy_env import RummyEnv  # Assuming your environment class is in rummy_env.py

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

    def __init__(self):
        # Initialization code for Rummy game environment
        self.hand_1 = []  # Player 1's hand
        self.hand_2 = []  # Player 2's hand
        self.discard_pile = []  # The discard pile
        self.deck = []  # The deck of cards
        self.current_turn = 1  # Who's turn is it
        self.rounds = 0  # Number of rounds played
        self.win_count = 0  # Player 1's win count
        self.loss_count = 0  # Player 1's loss count
        self.state = None  # Current state
        self.env = None  # External environment instance (if needed)
    
    def _check_win(self, hand):
        # Define the win condition (custom implementation for Rummy)
        pass
    
    def _evaluate_hand(self, hand):
        # Evaluate the hand (e.g., points or combinations in Rummy)
        pass
    
    def _get_observation(self):
        # Return current observation (state)
        return self.state

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Determine which player's hand is active based on current turn
        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2
        
        # Check if the active player has won
        if self._check_win(active_hand):
            terminated = True
            reward += 100  # Large reward for winning
            if self.current_turn == 1:
                self.win_count += 1
            else:
                self.loss_count += 1
        
        # If action is valid (player discards a card)
        if action < len(active_hand):  # Discard action
            discarded_card = active_hand.pop(action)
            self.discard_pile.append(discarded_card)
            reward += 0.5  # Small reward for discarding a card
        
        # If action is draw (player draws from the deck)
        elif action == len(active_hand):  # Draw from deck
            if len(self.deck) > 0:
                drawn_card = self.deck.pop()
                active_hand.append(drawn_card)
                reward += 0  # No immediate reward for drawing a card
            else:
                terminated = True  # Game ends if the deck is empty
        
        else:  # Invalid action (discarding something out of range)
            reward -= 10  # Large penalty for invalid actions

        # Evaluate the hand for points or combinations
        reward += self._evaluate_hand(active_hand)

        # If the player has won after evaluation
        if self._check_win(active_hand):
            terminated = True
            reward += 100  # Large reward for winning

        # Switch turns between players
        self.current_turn = 3 - self.current_turn
        self.rounds += 1  # Increment the round count

        # Get the new state/observation
        observation = self._get_observation()

        # Assume `self.env.step(action)` is another environment that needs to be called for some reason
        self.state, reward, done, info = self.env.step(action)  # If using external environment
        return self.state, reward, done, info  # Return 4 values

        # Return 5 values as required by the environment (observation, reward, done, truncated, additional info)
        return observation, reward, terminated, truncated, {}

    def reset(self):
        # Reset the environment to start a new game/round
        pass
    
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
        Evaluate the hand and assign a reward based on advanced strategies.
        """
        reward = 0

        # Track occurrences of pure sequences and valid sets
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

        # Log metrics for tracking
        self.metrics["pure_sequences"] += 1 if has_pure_sequence else 0
        self.metrics["valid_sets"] += 1 if has_valid_combinations else 0

        # Pure sequence check
        if has_pure_sequence:
            reward += 50

        # Valid combinations check
        if has_valid_combinations:
            reward += 30

        # High-point cards penalty
        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
            if card != 'joker' and card[0] in high_point_cards:
                reward -= 2  # Penalize holding high-point cards

        # Discarding high-point cards reward
        if self.current_turn == 1:
            last_discard = self.discard_pile[-1] if self.discard_pile else None
            if last_discard and last_discard != 'joker' and last_discard[0] in high_point_cards:
                reward += 5  # Reward discarding high-point cards
                
        # Penalize for excessive cards
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
    # Inside optimize_hyperparameters function
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir,
            batch_size=batch_size, n_steps=1024, learning_rate=learning_rate,
            gamma=gamma, exploration_fraction=exploration_fraction)

    # Training the model with timeout limit
    try:
        model.learn(total_timesteps=5000)  # Cap total training time
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


class MetricsLogger:
    def __init__(self):
        self.win_count = 0
        self.loss_count = 0
        self.total_rewards = []
        self.pure_sequences = 0
        self.valid_sets = 0
        self.actions = []
        self.rewards = []
        self.opponent_penalties = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0

    def log_game(self, total_reward, metrics, actions, rewards, won, opponent_penalties):
        self.total_rewards.append(total_reward)
        self.pure_sequences += metrics['pure_sequences']
        self.valid_sets += metrics['valid_sets']
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.opponent_penalties += opponent_penalties
        self.pure_sequences += metrics['pure_sequences']  # Correctly log pure sequences
        self.valid_sets += metrics['valid_sets']  # Correctly log valid sets

        if won:
            self.win_count += 1
        else:
            self.loss_count += 1

    def log_action_type(self, action, is_exploration):
        if is_exploration:
            self.exploration_actions += 1
        else:
            self.exploitation_actions += 1

    def calculate_metrics(self):
        win_rate = self.win_count / (self.win_count + self.loss_count) * 100
        avg_reward = np.mean(self.total_rewards)
        action_reward_corr = np.corrcoef(self.actions, self.rewards)[0, 1] if len(self.actions) > 1 else 0
        exploration_exploitation_ratio = self.exploration_actions / (self.exploitation_actions + 1e-5)

        return {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'pure_sequences_per_game': self.pure_sequences / len(self.total_rewards),
            'valid_sets_per_game': self.valid_sets / len(self.total_rewards),
            'action_reward_corr': action_reward_corr,
            'opponent_penalties': self.opponent_penalties / len(self.total_rewards),
            'exploration_exploitation_ratio': exploration_exploitation_ratio
        }

# Evaluation function with metric tracking
def evaluate_model(model_path, num_games=10):
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])
    model = PPO.load(model_path)
    metrics_logger = MetricsLogger()

    for game in range(num_games):
        obs = env.reset()
        terminated = False
        total_reward = 0
        actions = []
        rewards = []
        metrics = {'pure_sequences': 0, 'valid_sets': 0}
        opponent_penalties = 0

        while not terminated:
            action, _ = model.predict(obs)
            is_exploration = np.random.rand() < 0.1  # Example: Exploration with 10% probability
            metrics_logger.log_action_type(action[0], is_exploration)
            actions.append(action[0])
            
            # Unpack correctly for vectorized environment (4 values)
            obs, reward, terminated, truncated = env.step(action)
            total_reward += reward[0]  # Adjusted for the returned reward structure
            rewards.append(reward[0])
            
            # Remove references to 'info' since it's no longer returned
            # If you need to track pure sequences or valid sets, ensure they are handled elsewhere in your environment

        won = total_reward > 0
        metrics_logger.log_game(total_reward, metrics, actions, rewards, won, opponent_penalties)

    env.close()
    return metrics_logger.calculate_metrics()

# Training function with TensorBoard logging
def train_model(total_timesteps=5000, log_dir="./logs/rummy_ai/"):
    # Create directory for logging
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Create environment (only once)
    env = DummyVecEnv([lambda: RummyEnv(max_rounds=100)])

    # Create model (only once)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Exploration decay parameters
    exploration_rate = 1.0  # Start with full exploration
    min_exploration_rate = 0.01  # Minimum exploration rate after decay
    decay_rate = 0.995  # Decay rate for exploration

    # Training loop with exploration rate decay
    timestep_per_iteration = 2048  # Number of timesteps per iteration
    total_iterations = 0  # Keep track of iterations
    total_steps = 0  # Keep track of total timesteps

    # Training loop
    while total_steps < total_timesteps:
        # Update exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * decay_rate)

        # Perform a batch of training steps
        model.learn(total_timesteps=timestep_per_iteration)  # Train for a batch of steps
        total_iterations += 1  # Increment iterations
        total_steps += timestep_per_iteration  # Increment total timesteps

        # After training, evaluate the model and log metrics
        metrics = evaluate_model("rummy_ai_model", num_games=10)
        writer.add_scalar("Win Rate", metrics['win_rate'], total_steps)
        writer.add_scalar("Avg Reward", metrics['avg_reward'], total_steps)
        writer.add_scalar("Pure Sequences/Game", metrics['pure_sequences_per_game'], total_steps)
        writer.add_scalar("Valid Sets/Game", metrics['valid_sets_per_game'], total_steps)
        writer.add_scalar("Action-Reward Correlation", metrics['action_reward_corr'], total_steps)
        writer.add_scalar("Opponent Penalties/Game", metrics['opponent_penalties'], total_steps)
        writer.add_scalar("Exploration-Exploitation Ratio", metrics['exploration_exploitation_ratio'], total_steps)

        # Log iterations and total_timesteps to TensorBoard
        writer.add_scalar("Iterations", total_iterations, total_steps)
        writer.add_scalar("Total Timesteps", total_steps, total_steps)

        # Optionally log exploration rate to TensorBoard (optional but useful)
        writer.add_scalar("Exploration Rate", exploration_rate, total_steps)

    # Save the model after training
    model.save("rummy_ai_model")
    writer.close()


# Main execution
if __name__ == "__main__":
    train_model(total_timesteps=5000)
    metrics = evaluate_model("rummy_ai_model", num_games=50)

    print("Final Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
