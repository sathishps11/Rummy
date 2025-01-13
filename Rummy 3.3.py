import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import optuna
from stable_baselines3 import PPO
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn


class RummyEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, max_rounds=100, num_players=2):
        super().__init__()
        self.max_rounds = max_rounds
        self.num_players = num_players

        # Define Gym-compatible spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        self.action_space = spaces.Discrete(54)

        # Initialize game state
        self.agents = [f"player_{i}" for i in range(1, num_players + 1)]
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.discard_pile = []
        self.hands = {}
        self.deck = []
        self.rounds = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Call parent class reset if necessary
        self.rounds = 0
        self.deck = self._create_deck()
        random.shuffle(self.deck)

        self.hands = {agent: [self.deck.pop() for _ in range(13)] for agent in self.agents}
        self.discard_pile = [self.deck.pop()]
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}

        observation = self.observe(self.agent_selection)
        return observation, {}

    def observe(self, agent):
        """
        Return the observation for the given agent.
        """
        hand = self.hands[agent]
        obs = np.zeros(54, dtype=np.float32)

        # Encode the agent's hand into the observation vector
        for card in hand:
            obs[self._card_to_index(card)] += 1

    # Add metadata: normalized deck size and rounds
        max_deck_size = 54  # Standard deck size (52 cards + 2 jokers)
        deck_ratio = len(self.deck) / max_deck_size
        round_ratio = self.rounds / self.max_rounds
        obs = np.append(obs, [deck_ratio, round_ratio])

    # Ensure observation matches the declared observation space
        obs = obs.astype(np.float32)

    # Debug: Print the observation and metadata
        print(f"Generated observation: {obs}")
        print(f"Deck ratio: {deck_ratio}, Round ratio: {round_ratio}")

    # Ensure that the observation has the correct shape and values within range
        if obs.shape != (56,):  
           raise ValueError(f"Generated observation {obs} does not match the expected shape (56,)")  
    
        if not (obs.min() >= 0 and obs.max() <= 1):  
           raise ValueError(f"Generated observation {obs} has values outside the range [0, 1]")  

        if not self.observation_space.contains(obs):
           raise ValueError(f"Generated observation {obs} does not match the observation space {self.observation_space}")

        return obs


    def step(self, action):
        agent = self.agent_selection
        if self.dones[agent]:
           return self.observe(agent), 0, True, {}

        hand = self.hands[agent]
        reward = 0

        if action < len(hand):  # Discard action
           discarded_card = hand.pop(action)
           self.discard_pile.append(discarded_card)
           reward += 0.5
        elif action == len(hand):  # Draw action
             if self.deck:
                hand.append(self.deck.pop())
             else:
                self.dones = {a: True for a in self.agents}  # End game if deck is empty
        else:
            reward -= 10  # Invalid action penalty

    # Check winning condition
        if self._check_win(hand):
           reward += 100
           self.dones = {a: True for a in self.agents}

           self.rounds += 1
        if self.rounds >= self.max_rounds:
           self.dones = {a: True for a in self.agents}

        done = all(self.dones.values())
        next_agent_index = (self.agents.index(agent) + 1) % len(self.agents)
        self.agent_selection = self.agents[next_agent_index]
        obs = self.observe(self.agent_selection)
        return obs, reward, done, {}

        def observe(self, agent):
            """
            Return the observation for the given agent.
            """
            hand = self.hands[agent]
            obs = np.zeros(54, dtype=np.float32)
    
            # Encode the agent's hand into the observation vector
            for card in hand:
                obs[self._card_to_index(card)] += 1

            # Add metadata: normalized deck size and rounds
            max_deck_size = 54  # Standard deck size (52 cards + 2 jokers)
            obs = np.append(obs, [len(self.deck) / max_deck_size, self.rounds / self.max_rounds])

            # Ensure observation matches the declared observation space
            assert self.observation_space.contains(obs), f"Observation {obs} is not valid in the observation space!"

            return obs



    def render(self, mode="human"):
        """
        Render the current game state.
        """
        print(f"Current turn: {self.agent_selection}")
        for agent in self.agents:
            print(f"{agent}'s hand: {self.hands[agent]}")
        print(f"Discard pile: {self.discard_pile}")
        print(f"Rounds completed: {self.rounds}")

    def _create_deck(self):
        """
        Create a standard Rummy deck with 52 cards and 2 jokers.
        """
        suits = ['♥', '♦', '♣', '♠']  # Hearts, Diamonds, Clubs, Spades
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2

    def _card_to_index(self, card):
        """
        Convert a card to an index for the observation vector.
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

        self.metrics["pure_sequences"] += 1 if has_pure_sequence else 0
        self.metrics["valid_sets"] += 1 if has_valid_combinations else 0

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

    def _predict_opponent_strategy(self):
        """
        Analyze the discard pile and opponents' actions to predict their needs.
        """
        opponent_discard_tally = Counter(card[0] for card in self.discard_pile if card != 'joker')
        likely_needed = opponent_discard_tally.most_common(3)  # Top 3 most discarded ranks
        return [card for card, _ in likely_needed]


def make_rummy_env(max_rounds=100):
    env = RummyEnv(max_rounds=max_rounds)
    return env

# Wrap the environment for Stable-Baselines3
env = DummyVecEnv([lambda: make_rummy_env()])

# Using DummyVecEnv for a single environment
env = DummyVecEnv([lambda: make_rummy_env()])

# Or, if you want to use multiple environments in parallel:
# env = SubprocVecEnv([lambda: make_rummy_env() for _ in range(4)])




class RewardLoggerCallback(BaseCallback):
    def __init__(self, writer, live_plot, verbose=0):
        """
        Callback to log rewards to TensorBoard and update live visuals during training.
        
        Args:
            writer (SummaryWriter): TensorBoard SummaryWriter for logging metrics.
            live_plot (LivePlot): Instance of the LivePlot class for real-time visualization.
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.writer = writer
        self.timesteps = []
        self.rewards = []
        self.live_plot = live_plot

    def _on_step(self) -> bool:
        """
        Logs the rewards, updates live plot, and writes metrics to TensorBoard at intervals.
        
        Returns:
            bool: Whether the training should continue.
        """
        # Get current timestep and reward
        current_timestep = self.num_timesteps
        rewards = self.locals.get("rewards", None)
        
        # Handle missing rewards key in locals
        if rewards is not None:
            reward = rewards[0]
            self.timesteps.append(current_timestep)
            self.rewards.append(reward)

            # Debugging: Print current state (if verbose is enabled)
            if self.verbose > 0:
                print(f"Timestep: {current_timestep}, Reward: {reward}")

            # Update live plot and TensorBoard every 1000 timesteps
            if current_timestep % 1000 == 0:
                avg_reward = np.mean(self.rewards[-100:])  # Compute average of last 100 rewards
                self.writer.add_scalar("Average Reward", avg_reward, current_timestep)
                
                if self.verbose > 0:
                    print(f"Updating plot at timestep {current_timestep}, Avg Reward: {avg_reward:.2f}")
                
                self.live_plot.update([self.timesteps, self.rewards])

        return True

class LiveDashboard:
    def __init__(self):
        """
        Initializes the live dashboard with multiple subplots.
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.ion()

        # Subplots
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

        # Data containers
        self.timesteps = []
        self.rewards = []
        self.avg_rewards = []
        self.wins = []
        self.losses = []
        self.pure_sequences = []
        self.valid_sets = []
        self.action_counts = Counter()

    def update(self, timestep, reward, avg_reward, win_count, loss_count, metrics, action):
        """
        Updates the dashboard with new data.
        """
        # Update data
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.wins.append(win_count)
        self.losses.append(loss_count)
        self.pure_sequences.append(metrics["pure_sequences"])
        self.valid_sets.append(metrics["valid_sets"])
        self.action_counts[action] += 1

        # Update reward plot
        self.reward_line.set_data(self.timesteps, self.rewards)
        self.reward_avg_line.set_data(self.timesteps, self.avg_rewards)
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()

        # Update win/loss trend
        self.win_line.set_data(range(len(self.wins)), self.wins)
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.win_ax.relim()
        self.win_ax.autoscale_view()

        # Update action selection frequency
        if self.action_bars:
            for bar, count in zip(self.action_bars, self.action_counts.values()):
                bar.set_height(count)
        else:
            self.action_bars = self.action_ax.bar(self.action_counts.keys(), self.action_counts.values())
        self.action_ax.relim()
        self.action_ax.autoscale_view()

        # Update metric trends
        self.pure_seq_line.set_data(self.timesteps, self.pure_sequences)
        self.valid_sets_line.set_data(self.timesteps, self.valid_sets)
        self.metric_ax.relim()
        self.metric_ax.autoscale_view()

        # Redraw plots
        plt.pause(0.01)
        plt.draw()


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


def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int('num_envs', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['constant', 'exponential_decay'])

    # Create the environment
    env = SubprocVecEnv([lambda: make_rummy_env() for _ in range(num_envs)])


    # Initialize PPO model
    model = PPO("MlpPolicy", env, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate, verbose=0)

    # Implement learning rate scheduler if 'exponential_decay' is chosen
    if lr_scheduler == 'exponential_decay':
        def lr_schedule(_):
            return learning_rate * (0.95 ** np.arange(1000)[_])  # Update learning rate exponentially

        model.set_parameters({'learning_rate': lr_schedule})

    # Train the model
    model.learn(total_timesteps=5000)

    # Evaluate the model (you need to define evaluate_model separately)
    win_rate = evaluate_model(model)

    # Close environment
    env.close()

    return win_rate

def optimize_rewards(trial):
    w_pure_seq = trial.suggest_int("w_pure_seq", 25, 35)
    w_valid_set = trial.suggest_int("w_valid_set", 45, 55)
    w_high_point_penalty = trial.suggest_int("w_high_point_penalty", -6, -4)
    w_discard_high = trial.suggest_int("w_discard_high", 7, 11)
    w_stall_penalty = trial.suggest_int("w_stall_penalty", -7, -4)
    w_joker_usage = trial.suggest_int("w_joker_usage", 10, 15)

    # Rest of the optimization code


    # Custom reward evaluation logic
    def custom_evaluate_hand(self, hand):
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

    # Update the environment's reward function dynamically
    RummyEnv._evaluate_hand = custom_evaluate_hand

    # Create the environment
    num_envs = trial.suggest_int("num_envs", 2, 4)  # Parallel environments
    env = SubprocVecEnv([lambda: RummyEnv(max_rounds=100) for _ in range(num_envs)])

    # Model configuration
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    model = PPO("MlpPolicy", env, verbose=0, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)

    # Train the model
    model.learn(total_timesteps=5000)

    # Evaluate the model using win rate as the metric
    win_rate = evaluate_model(model, num_games=50)
    env.close()
    return win_rate

class LivePlot:
    def __init__(self):
        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.reward_history = []  # Keep it as a simple list
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Live Training Reward Plot")
        plt.ion()  # Enable interactive mode

    def update(self, reward):
        try:
            # Ensure reward is a 1D array or scalar
            reward = np.array(reward).flatten()

            # Check if reward is 1D and append properly
            if reward.ndim != 1:
                print(f"Error: Expected 1D array, but got {reward.ndim}D.")
                return

            # Append the reward if it's a 1D array or scalar
            if reward.size == 1:  # It's a scalar
                self.reward_history.append(reward[0])
            else:  # It's a 1D array
                self.reward_history.extend(reward)  # Extend with values if it's an array

            # Debugging: print the shape of reward_history
            print(f"reward_history length: {len(self.reward_history)}")

            # Clear the axis to update it
            self.ax.clear()

            # Plot the rewards history
            self.ax.plot(self.reward_history)

            self.ax.set_xlabel("Training Steps")
            self.ax.set_ylabel("Average Reward")
            self.ax.set_title("Live Training Progress")

            plt.draw()  # Redraw the plot
            plt.pause(0.01)  # Pause to allow the plot to update

        except Exception as e:
            print(f"Error during update: {e}")

    def show_plot(self):
        """ Call plt.show() to ensure the plot is displayed. """
        plt.show()


def train_model(total_timesteps=5000, num_envs=2):
    """
    Train the model with the given parameters.
    """
    # Create log directory
    log_dir = "./ppo_rummy_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir + "custom_metrics/")

    # Debug: Use DummyVecEnv first to ensure no multiprocessing issues
    from pettingzoo.utils.conversions import parallel_wrapper_fn

    # Wrap RummyEnv with parallel_wrapper_fn
    parallel_env = parallel_wrapper_fn(lambda: RummyEnv(max_rounds=100))

    # Use the wrapped environment in DummyVecEnv
    env = DummyVecEnv([lambda: make_rummy_env()])


    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

    # Initialize live plot and reward logger callback
    try:
        live_plot = LivePlot()  # Ensure LivePlot is properly initialized
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LivePlot: {e}")

    reward_logger = RewardLoggerCallback(writer, live_plot)

    # Train the model
    try:
        model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    except Exception as e:
        print(f"Model training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}")

    # Save the model
    model.save("rummy_ai_model")

    # Close writer and finalize plot
    writer.close()
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot

# Main Execution
from pettingzoo.utils.conversions import parallel_wrapper_fn
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: make_rummy_env()])




if __name__ == "__main__":
    # Wrap the Rummy environment to make it Gym-compatible
    parallel_env = parallel_wrapper_fn(make_rummy_env)
    
    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize", study_name="Rummy Reward Optimization")

    # Optimize rewards (define optimize_rewards function separately)
    study.optimize(optimize_rewards, n_trials=2)

    # Retrieve the best parameters
    num_envs = study.best_params.get("num_envs", 1)  # Default to 1 if 'num_envs' is missing

    # Create the SubprocVecEnv with Gym-compatible wrapped environments
    env = SubprocVecEnv([lambda: parallel_env for _ in range(num_envs)])

    print("Best parameters:", study.best_params)

    # Train the model using optimized hyperparameters
    train_model(total_timesteps=5000, num_envs=num_envs)

