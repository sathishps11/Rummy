import random
from collections import Counter
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

class MultiAgentRummyEnv(ParallelEnv):
    """
    Multi-Agent Rummy Environment
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, max_rounds=100, num_players=2, use_jokers=True):
        """
        Initialize the Multi-Agent Rummy environment.
        """
        self.max_rounds = max_rounds
        self.num_players = num_players
        self.use_jokers = use_jokers

        # Define observation and action spaces
        self.action_space = {f"player_{i}": spaces.Discrete(54) for i in range(num_players)}
        self.observation_space = {f"player_{i}": spaces.Box(low=0, high=1, shape=(114,), dtype=np.float32) for i in range(num_players)}

        # Initialize game state
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.hands = {}
        self.deck = []
        self.discard_pile = []
        self.rounds = 0

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.rounds = 0
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.hands = {agent: [self.deck.pop() for _ in range(13)] for agent in self.agents}
        self.discard_pile = [self.deck.pop()]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.agent_selection = self.agents[0]
        return self.observe_all()

    def step(self, actions):
        """
        Execute actions for all agents.
        """
        for agent, action in actions.items():
            if self.dones[agent]:
                continue

            # Action logic (discard/draw)
            hand = self.hands[agent]
            if action < len(hand):  # Discard
                discarded_card = hand.pop(action)
                self.discard_pile.append(discarded_card)
            elif action == len(hand):  # Draw
                if self.deck:
                    hand.append(self.deck.pop())

            self.rewards[agent] = self._evaluate_hand(hand)
            self.dones[agent] = self._check_win(hand) or self.rounds >= self.max_rounds

        # Increment rounds and update done status
        self.rounds += 1
        all_done = all(self.dones.values())
        return self.observe_all(), self.rewards, {agent: all_done for agent in self.agents}, {}

    def observe_all(self):
        """
        Return observations for all agents.
        """
        return {agent: self.observe(agent) for agent in self.agents}

    def observe(self, agent):
        """
        Return observation for a single agent.
        """
        hand = self.hands[agent]
        obs = np.zeros(54, dtype=np.float32)
        for card in hand:
            obs[self._card_to_index(card)] += 1

        deck_ratio = len(self.deck) / 54
        round_ratio = self.rounds / self.max_rounds
        discard_obs = np.zeros(54, dtype=np.float32)
        for card in self.discard_pile[-3:]:
            discard_obs[self._card_to_index(card)] += 1

        return np.concatenate([obs, [deck_ratio, round_ratio], discard_obs]).astype(np.float32)

    def render(self):
        """
        Render the current game state.
        """
        print(f"Round: {self.rounds}")
        print(f"Hands: {self.hands}")
        print(f"Deck size: {len(self.deck)}")
        print(f"Discard pile: {self.discard_pile}")

    def _create_deck(self):
        """
        Create a standard deck of cards (with or without jokers).
        """
        suits = ['♥', '♦', '♣', '♠']
        ranks = range(1, 14)
        deck = [(rank, suit) for suit in suits for rank in ranks]
        if self.use_jokers:
            deck += ['joker'] * 2
        return deck

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
        Check if the hand meets the win conditions.
        """
        return self._has_pure_sequence(hand) and self._has_valid_combinations(hand)

    def _evaluate_hand(self, hand):
        """
        Evaluate the current hand and assign a reward.
        """
        reward = 0
        if self._has_pure_sequence(hand):
            reward += 50
        if self._has_valid_combinations(hand):
            reward += 30
        return reward

    def _has_pure_sequence(self, hand):
        """
        Check if the hand contains a pure sequence.
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
        Check for valid sets or sequences in the hand.
        """
        rank_counts = Counter(card[0] for card in hand if card != 'joker')
        return sum(1 for count in rank_counts.values() if count >= 3) >= 2
    
    
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rummy_env import MultiAgentRummyEnv


class RummyTrainer:
    """
    Encapsulates training logic for AI agents in the Rummy environment.
    """

    def __init__(self, env_class, num_envs=4, log_dir="./logs", total_timesteps=100_000):
        """
        Initialize the trainer with the specified environment and training parameters.
        """
        self.env_class = env_class
        self.num_envs = num_envs
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps
        self.model = None

        # Create the logging directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up vectorized environments for parallel training
        self.env = SubprocVecEnv([self._make_env for _ in range(self.num_envs)])

    def _make_env(self):
        """
        Create a new instance of the environment for parallelization.
        """
        return self.env_class(max_rounds=100, num_players=2)

    def initialize_model(self, policy="MlpPolicy", batch_size=256, gamma=0.99, learning_rate=1e-3):
        """
        Initialize the PPO model with the specified hyperparameters.
        """
        self.model = PPO(
            policy,
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
        )

    def train(self, callback=None):
        """
        Train the PPO model using the environment.
        """
        if not self.model:
            raise ValueError("Model is not initialized. Call `initialize_model` first.")

        print("Starting training...")
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)
        self.save_model("trained_rummy_model")
        print("Training completed!")

    def save_model(self, model_name):
        """
        Save the trained model to the log directory.
        """
        save_path = os.path.join(self.log_dir, model_name)
        self.model.save(save_path)
        print(f"Model saved at: {save_path}")

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified path.
        """
        self.model = PPO.load(model_path, env=self.env)
        print(f"Model loaded from: {model_path}")

    def evaluate(self, num_games=50):
        """
        Evaluate the model by simulating a specified number of games.
        """
        if not self.model:
            raise ValueError("Model is not loaded or initialized. Train or load a model first.")

        env = self.env_class(max_rounds=100, num_players=2)
        wins = 0

        for _ in range(num_games):
            obs = env.reset()
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                actions = {}
                for agent in env.agents:
                    if not done[agent]:
                        action, _ = self.model.predict(obs[agent], deterministic=True)
                        actions[agent] = action

                obs, rewards, done, _ = env.step(actions)

            for reward in rewards.values():
                if reward > 0:  # Positive reward indicates a win
                    wins += 1

        win_rate = wins / (num_games * len(env.agents))
        print(f"Win Rate: {win_rate * 100:.2f}%")
        return win_rate

from collections import Counter

def create_deck(use_jokers=True):
    """
    Create a standard deck of Rummy cards, optionally including jokers.

    Args:
        use_jokers (bool): Whether to include jokers in the deck.

    Returns:
        list: A list of cards in the deck.
    """
    suits = ['♥', '♦', '♣', '♠']
    ranks = range(1, 14)
    deck = [(rank, suit) for suit in suits for rank in ranks]
    if use_jokers:
        deck += ['joker'] * 2
    return deck


def card_to_index(card):
    """
    Convert a card to its corresponding index for observation encoding.

    Args:
        card (tuple or str): A card represented as (rank, suit) or 'joker'.

    Returns:
        int: The index of the card in the observation space.
    """
    if card == 'joker':
        return 52
    rank, suit = card
    suit_offset = {'♥': 0, '♦': 13, '♣': 26, '♠': 39}[suit]
    return suit_offset + rank - 1


def has_pure_sequence(hand):
    """
    Check if the hand contains at least one pure sequence.

    Args:
        hand (list): List of cards in the hand.

    Returns:
        bool: True if a pure sequence exists, False otherwise.
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


def has_valid_combinations(hand):
    """
    Check if the hand contains at least two valid sets or sequences.

    Args:
        hand (list): List of cards in the hand.

    Returns:
        bool: True if the hand contains valid combinations, False otherwise.
    """
    rank_counts = Counter(card[0] for card in hand if card != 'joker')
    sets = sum(1 for count in rank_counts.values() if count >= 3)

    sequences = 0
    sorted_hand = sorted([card for card in hand if card != 'joker'], key=lambda x: (x[1], x[0]))
    temp_seq = []
    for card in sorted_hand:
        if temp_seq and card[1] == temp_seq[-1][1] and card[0] == temp_seq[-1][0] + 1:
            temp_seq.append(card)
        else:
            if len(temp_seq) >= 3:
                sequences += 1
            temp_seq = [card]

    if len(temp_seq) >= 3:
        sequences += 1

    return sets + sequences >= 2


def evaluate_hand(hand):
    """
    Evaluate a hand and calculate a reward based on the hand's validity.

    Args:
        hand (list): List of cards in the hand.

    Returns:
        int: Reward based on the hand's composition.
    """
    reward = 0
    if has_pure_sequence(hand):
        reward += 50
    if has_valid_combinations(hand):
        reward += 30
    return reward


def encode_observation(hand, deck_size, discard_pile, rounds, max_rounds):
    """
    Encode the game state into an observation vector.

    Args:
        hand (list): List of cards in the player's hand.
        deck_size (int): Number of cards remaining in the deck.
        discard_pile (list): Cards in the discard pile.
        rounds (int): Current round number.
        max_rounds (int): Total number of rounds.

    Returns:
        numpy.ndarray: Encoded observation vector.
    """
    import numpy as np

    # Encode the player's hand
    hand_obs = np.zeros(54, dtype=np.float32)
    for card in hand:
        hand_obs[card_to_index(card)] += 1

    # Encode the deck size and round number
    deck_ratio = deck_size / 54
    round_ratio = rounds / max_rounds

    # Encode the discard pile (last three cards)
    discard_obs = np.zeros(54, dtype=np.float32)
    for card in discard_pile[-3:]:
        discard_obs[card_to_index(card)] += 1

    return np.concatenate([hand_obs, [deck_ratio, round_ratio], discard_obs]).astype(np.float32)


import numpy as np


class RandomPlayer:
    """
    A baseline player that chooses actions randomly.
    """

    def choose_action(self, observation):
        """
        Select a random action from the available actions.

        Args:
            observation (numpy.ndarray): Encoded observation vector.

        Returns:
            int: Chosen action index.
        """
        action_space_size = 54  # Total actions (draw/discard)
        return np.random.randint(0, action_space_size)


class HeuristicPlayer:
    """
    A baseline player that uses a simple heuristic strategy to make decisions.
    """

    def choose_action(self, observation):
        """
        Select an action based on a simple heuristic:
        - Discard cards least likely to form a sequence or set.

        Args:
            observation (numpy.ndarray): Encoded observation vector.

        Returns:
            int: Chosen action index.
        """
        # Decode the observation to retrieve the hand encoding
        hand_obs = observation[:54]
        card_values = [(i, value) for i, value in enumerate(hand_obs) if value > 0]

        # Sort cards by their likelihood of forming a sequence or set
        # (Assumes that lower values are less likely to form combinations)
        sorted_cards = sorted(card_values, key=lambda x: x[1])

        # Discard the card with the lowest likelihood of forming combinations
        return sorted_cards[0][0] if sorted_cards else 0


class RuleBasedPlayer:
    """
    A more advanced baseline player that prioritizes creating sequences and sets.
    """

    def choose_action(self, observation):
        """
        Select an action by identifying cards that contribute to sequences or sets.

        Args:
            observation (numpy.ndarray): Encoded observation vector.

        Returns:
            int: Chosen action index.
        """
        # Decode the observation to retrieve the hand encoding
        hand_obs = observation[:54]
        deck_ratio = observation[54]  # Cards remaining in the deck
        round_ratio = observation[55]  # Current round number as a fraction

        # Identify cards that are part of sequences or sets
        card_values = [(i, value) for i, value in enumerate(hand_obs) if value > 0]
        potential_sequences = [card for card in card_values if self._is_part_of_sequence(card[0], hand_obs)]

        # If there are potential sequences, prioritize discarding non-sequence cards
        if potential_sequences:
            return potential_sequences[0][0]

        # Otherwise, discard the lowest-value card
        sorted_cards = sorted(card_values, key=lambda x: x[1])
        return sorted_cards[0][0] if sorted_cards else 0

    def _is_part_of_sequence(self, card_index, hand_obs):
        """
        Check if a card is part of a potential sequence.

        Args:
            card_index (int): Index of the card to check.
            hand_obs (numpy.ndarray): Encoded hand observation.

        Returns:
            bool: True if the card is part of a sequence, False otherwise.
        """
        suit_offset = card_index // 13 * 13
        rank = card_index % 13 + 1

        # Check adjacent ranks in the same suit
        left_neighbor = hand_obs[suit_offset + rank - 2] if rank > 1 else 0
        right_neighbor = hand_obs[suit_offset + rank] if rank < 13 else 0
        return left_neighbor > 0 or right_neighbor > 0

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class LivePlot:
    """
    Real-time plot for tracking rewards during training.
    """

    def __init__(self, title="Training Progress", xlabel="Steps", ylabel="Rewards"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.reward_history = []

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.reward_line, = self.ax.plot([], [], label="Rewards")
        self.ax.legend()
        plt.ion()  # Enable interactive mode

    def update(self, reward):
        """
        Update the plot with a new reward value.

        Args:
            reward (float): The new reward to add to the plot.
        """
        self.reward_history.append(reward)
        self.reward_line.set_data(range(len(self.reward_history)), self.reward_history)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def show(self):
        """
        Display the final plot.
        """
        plt.ioff()
        plt.show()


class TrainingDashboard:
    """
    A comprehensive dashboard for tracking multiple metrics during training.
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.ion()  # Enable interactive mode

        # Subplots for metrics
        self.reward_ax = self.axes[0, 0]
        self.reward_ax.set_title("Reward Trend")
        self.reward_ax.set_xlabel("Steps")
        self.reward_ax.set_ylabel("Rewards")
        self.reward_line, = self.reward_ax.plot([], [], label="Rewards")
        self.reward_avg_line, = self.reward_ax.plot([], [], label="Average Reward")
        self.reward_ax.legend()

        self.win_loss_ax = self.axes[0, 1]
        self.win_loss_ax.set_title("Win/Loss")
        self.win_loss_ax.set_xlabel("Episodes")
        self.win_loss_ax.set_ylabel("Count")
        self.win_line, = self.win_loss_ax.plot([], [], label="Wins")
        self.loss_line, = self.win_loss_ax.plot([], [], label="Losses")
        self.win_loss_ax.legend()

        self.action_ax = self.axes[1, 0]
        self.action_ax.set_title("Action Distribution")
        self.action_ax.set_xlabel("Action")
        self.action_ax.set_ylabel("Frequency")
        self.action_bars = None

        self.metric_ax = self.axes[1, 1]
        self.metric_ax.set_title("Gameplay Metrics")
        self.metric_ax.set_xlabel("Steps")
        self.metric_ax.set_ylabel("Count")
        self.metric_pure_seq_line, = self.metric_ax.plot([], [], label="Pure Sequences")
        self.metric_valid_sets_line, = self.metric_ax.plot([], [], label="Valid Sets")
        self.metric_ax.legend()

        # Data storage
        self.steps = []
        self.rewards = []
        self.avg_rewards = []
        self.wins = []
        self.losses = []
        self.action_counts = Counter()
        self.pure_sequences = []
        self.valid_sets = []

    def update(self, step, reward, avg_reward, wins, losses, action=None, pure_sequences=0, valid_sets=0):
        """
        Update the dashboard with new training data.

        Args:
            step (int): Current training step.
            reward (float): Current reward value.
            avg_reward (float): Average reward value.
            wins (int): Number of wins.
            losses (int): Number of losses.
            action (int): Most recent action taken.
            pure_sequences (int): Number of pure sequences formed.
            valid_sets (int): Number of valid sets formed.
        """
        # Update metrics
        self.steps.append(step)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.wins.append(wins)
        self.losses.append(losses)
        self.pure_sequences.append(pure_sequences)
        self.valid_sets.append(valid_sets)
        if action is not None:
            self.action_counts[action] += 1

        # Update reward plot
        self.reward_line.set_data(self.steps, self.rewards)
        self.reward_avg_line.set_data(self.steps, self.avg_rewards)
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()

        # Update win/loss plot
        self.win_line.set_data(range(len(self.wins)), self.wins)
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.win_loss_ax.relim()
        self.win_loss_ax.autoscale_view()

        # Update action distribution plot
        if self.action_bars:
            for bar, count in zip(self.action_bars, self.action_counts.values()):
                bar.set_height(count)
        else:
            self.action_bars = self.action_ax.bar(self.action_counts.keys(), self.action_counts.values())
        self.action_ax.relim()
        self.action_ax.autoscale_view()

        # Update gameplay metrics plot
        self.metric_pure_seq_line.set_data(self.steps, self.pure_sequences)
        self.metric_valid_sets_line.set_data(self.steps, self.valid_sets)
        self.metric_ax.relim()
        self.metric_ax.autoscale_view()

        plt.pause(0.01)

    def show(self):
        """
        Display the final dashboard.
        """
        plt.ioff()
        plt.show()

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import json


class RewardLoggerCallback(BaseCallback):
    """
    Logs rewards during training for visualization and debugging.
    """

    def __init__(self, log_dir="./logs", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.reward_log_path = os.path.join(self.log_dir, "reward_log.json")
        self.rewards = []

        # Create the log directory if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each training step.
        Logs the reward for visualization.
        """
        reward = self.locals.get("rewards", None)
        if reward is not None:
            avg_reward = np.mean(reward)
            self.rewards.append(avg_reward)

        return True

    def on_training_end(self):
        """
        Called at the end of training.
        Saves logged rewards to a JSON file.
        """
        with open(self.reward_log_path, "w") as f:
            json.dump(self.rewards, f)
        print(f"Reward logs saved to {self.reward_log_path}")


class LiveVisualizationCallback(BaseCallback):
    """
    Updates live plots and dashboards during training.
    """

    def __init__(self, visualizer, verbose=0):
        """
        Args:
            visualizer (object): An instance of `LivePlot` or `TrainingDashboard`.
        """
        super().__init__(verbose)
        self.visualizer = visualizer

    def _on_step(self) -> bool:
        """
        Called at each training step.
        Updates the visualizer with new metrics.
        """
        reward = self.locals.get("rewards", None)
        timestep = self.num_timesteps
        avg_reward = np.mean(reward) if reward is not None else 0

        # Update visualizer
        if isinstance(self.visualizer, TrainingDashboard):
            metrics = self.locals.get("metrics", {})
            action = self.locals.get("actions", [0])[0]
            wins = metrics.get("wins", 0)
            losses = metrics.get("losses", 0)
            pure_sequences = metrics.get("pure_sequences", 0)
            valid_sets = metrics.get("valid_sets", 0)
            self.visualizer.update(timestep, avg_reward, avg_reward, wins, losses, action, pure_sequences, valid_sets)
        elif isinstance(self.visualizer, LivePlot):
            self.visualizer.update(avg_reward)

        return True


class EnhancedReplayLoggerCallback(BaseCallback):
    """
    Logs detailed replay data for training analysis.
    """

    def __init__(self, log_dir="./logs", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.replay_log_path = os.path.join(self.log_dir, "replay_log.json")
        self.replay_data = []

        # Create the log directory if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each training step.
        Logs observations, actions, rewards, and dones.
        """
        obs = self.locals.get("obs", None)
        action = self.locals.get("actions", None)
        reward = self.locals.get("rewards", None)
        done = self.locals.get("dones", None)

        if obs is not None and action is not None:
            self.replay_data.append({
                "obs": obs.tolist() if hasattr(obs, "tolist") else obs,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": float(reward) if isinstance(reward, (int, float)) else reward,
                "done": done.tolist() if hasattr(done, "tolist") else done,
            })

        return True

    def on_training_end(self):
        """
        Called at the end of training.
        Saves logged replay data to a JSON file.
        """
        with open(self.replay_log_path, "w") as f:
            json.dump(self.replay_data, f)
        print(f"Replay logs saved to {self.replay_log_path}")

from rummy_env import MultiAgentRummyEnv
from rummy_ai import RummyTrainer
from rummy_visualizer import TrainingDashboard
from rummy_callbacks import RewardLoggerCallback, LiveVisualizationCallback
from rummy_baseline import RandomPlayer, HeuristicPlayer, RuleBasedPlayer


def train_ai_agent(total_timesteps=50_000, num_envs=4, log_dir="./logs"):
    """
    Train an AI agent in the Rummy environment.
    """
    print("=== Initializing Training Environment ===")
    trainer = RummyTrainer(env_class=MultiAgentRummyEnv, num_envs=num_envs, log_dir=log_dir, total_timesteps=total_timesteps)

    # Initialize the model
    print("=== Initializing PPO Model ===")
    trainer.initialize_model(policy="MlpPolicy", batch_size=128, gamma=0.99, learning_rate=1e-4)

    # Set up visualization and logging
    dashboard = TrainingDashboard()
    reward_logger = RewardLoggerCallback(log_dir=log_dir)
    visualization_callback = LiveVisualizationCallback(visualizer=dashboard)

    # Train the model
    print("=== Starting Training ===")
    trainer.train(callback=[reward_logger, visualization_callback])

    # Save the model
    print("=== Training Complete! ===")
    return trainer


def evaluate_trained_model(trainer, num_games=20):
    """
    Evaluate a trained model by simulating games.
    """
    print("=== Evaluating Trained Model ===")
    win_rate = trainer.evaluate(num_games=num_games)
    print(f"Win Rate: {win_rate * 100:.2f}%")
    return win_rate


def benchmark_baseline_players(num_games=10):
    """
    Benchmark baseline players in the Rummy environment.
    """
    print("=== Benchmarking Baseline Players ===")
    env = MultiAgentRummyEnv(max_rounds=100, num_players=2)
    random_player = RandomPlayer()
    heuristic_player = HeuristicPlayer()
    rule_based_player = RuleBasedPlayer()

    baseline_results = {"RandomPlayer": 0, "HeuristicPlayer": 0, "RuleBasedPlayer": 0}

    for _ in range(num_games):
        obs = env.reset()
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if agent == "player_0":
                    actions[agent] = random_player.choose_action(obs[agent])
                elif agent == "player_1":
                    actions[agent] = heuristic_player.choose_action(obs[agent])
                else:
                    actions[agent] = rule_based_player.choose_action(obs[agent])

            obs, rewards, done, _ = env.step(actions)

        for agent, reward in rewards.items():
            if reward > 0:
                if agent == "player_0":
                    baseline_results["RandomPlayer"] += 1
                elif agent == "player_1":
                    baseline_results["HeuristicPlayer"] += 1
                else:
                    baseline_results["RuleBasedPlayer"] += 1

    # Calculate win rates
    for player, wins in baseline_results.items():
        win_rate = wins / (num_games * len(env.agents)) * 100
        print(f"{player} Win Rate: {win_rate:.2f}%")
    return baseline_results


def run():
    """
    Run the main script for training and evaluation.
    """
    # Train the AI agent
    trainer = train_ai_agent(total_timesteps=50_000, num_envs=4, log_dir="./logs")

    # Evaluate the trained model
    evaluate_trained_model(trainer, num_games=20)

    # Benchmark baseline players
    benchmark_baseline_players(num_games=10)


if __name__ == "__main__":
    run()

