from pettingzoo import AECEnv  # Keep this import as it’s correct
from gymnasium import spaces
import numpy as np
import random
from collections import Counter
from pettingzoo.classic import rummy


class MultiAgentRummyEnv(AECEnv):
    """
    Multi-agent Rummy environment adhering to PettingZoo standards.
    """

    metadata = {"render_modes": ["human"], "name": "multi_agent_rummy"}

    def __init__(self, max_rounds=100, num_players=2):
        super().__init__()
        self.max_rounds = max_rounds
        self.num_players = num_players

        # Define action and observation spaces
        self.action_space = spaces.Discrete(54)  # 54 actions: discard/draw
        self.observation_space = spaces.Box(low=0, high=1, shape=(168,), dtype=np.float32)

        # Initialize the game state
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}
        self.hands = {}
        self.deck = []
        self.discard_pile = []
        self.rounds = 0

        # Metrics to track game performance
        self.metrics = {
            "pure_sequences": 0,
            "valid_sets": 0,
            "blocks": 0,
            "high_point_penalty": 0,
            "help_opponent": 0,
        }

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
        self.truncateds = {agent: False for agent in self.agents}

        return self.observe_all()

    def observe_all(self):
        """
        Generate observations for all agents.
        """
        return {agent: self.observe(agent) for agent in self.agents}

    def observe(self, agent):
        """
        Generate an observation for a specific agent.
        """
        hand = self.hands[agent]
        obs = np.zeros(54, dtype=np.float32)

        # Encode the player's hand
        for card in hand:
            obs[self._card_to_index(card)] += 1

        # Add discard pile and game metadata
        discard_obs = np.zeros(54, dtype=np.float32)
        for card in self.discard_pile[-3:]:
            discard_obs[self._card_to_index(card)] += 1

        deck_ratio = len(self.deck) / 54
        round_ratio = self.rounds / self.max_rounds

        obs = np.concatenate([obs, [deck_ratio, round_ratio], discard_obs])
        if obs.size < 168:
            obs = np.pad(obs, (0, 168 - obs.size), constant_values=0)
        return obs

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            if self.dones[agent]:
                continue

            hand = self.hands[agent]

            # Process discard or draw actions
            if action < len(hand):  # Discard
                discarded_card = hand.pop(action)
                self.discard_pile.append(discarded_card)
                rewards[agent] = self._evaluate_hand(hand)
                self._affect_opponents(discarded_card)
            elif action == len(hand):  # Draw
                if self.deck:
                    hand.append(self.deck.pop())
                else:
                    infos[agent]["error"] = "Deck is empty"

            self.dones[agent] = self._check_win(hand)

        self.rounds += 1
        if self.rounds >= self.max_rounds:
            for agent in self.agents:
                self.truncateds[agent] = True

        observations = self.observe_all()
        return observations, rewards, self.dones, self.truncateds, infos

    def render(self, mode="human"):
        print(f"Round: {self.rounds}, Discard Pile: {self.discard_pile}")

    def _create_deck(self):
        suits = ["♥", "♦", "♣", "♠"]
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ["joker"] * 2

    def _card_to_index(self, card):
        if card == "joker":
            return 52
        rank, suit = card
        suit_offset = {"♥": 0, "♦": 13, "♣": 26, "♠": 39}[suit]
        return suit_offset + rank - 1

    def _evaluate_hand(self, hand):
        reward = 0
        if self._has_pure_sequence(hand):
            reward += 50
        if self._has_valid_combinations(hand):
            reward += 30
        return reward

    def _check_win(self, hand):
        return self._has_pure_sequence(hand) and self._has_valid_combinations(hand)

    def _has_pure_sequence(self, hand):
        sorted_hand = sorted([card for card in hand if card != "joker"], key=lambda x: (x[1], x[0]))
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
        rank_counts = Counter(card[0] for card in hand if card != "joker")
        return sum(1 for count in rank_counts.values() if count >= 3) >= 2

    def _affect_opponents(self, discarded_card):
        for opponent in self.agents:
            if not self.dones[opponent] and discarded_card in self.hands[opponent]:
                self.rewards[opponent] += 10


# Remove the second _evaluate_hand definition
    def _evaluate_hand(self, hand):
        """
        Evaluate the current player's hand with strategic rewards.
        """
        reward = 0
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

    # Reward for forming pure sequences and valid sets
        if has_pure_sequence:
           reward += 50
        if has_valid_combinations:
           reward += 30

    # Penalties for enabling opponents
        high_point_cards = [10, 11, 12, 13, 1]
        for card in hand:
           if card != 'joker' and card[0] in high_point_cards:
              reward -= 2

    # Incentive for discarding high-point cards
        if len(self.discard_pile) > 0:
           last_discard = self.discard_pile[-1]
           if last_discard[0] in high_point_cards:
              reward += 5

    # Bluffing incentive: Successfully trick opponents
        bluff_success = random.choice([True, False])  # Replace with bluff logic
        if bluff_success:
           reward += 20

        return reward



#Module 2: Reward Logger and Callbacks 
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    """
    Callback to log rewards and update a live dashboard or plot.
    """

    def __init__(self, writer, plot_dashboard):
        super().__init__()
        self.writer = writer
        self.plot_dashboard = plot_dashboard

    def step(self, actions):
        # Ensure actions are in dictionary format
        if not isinstance(actions, dict):
            if isinstance(actions, (np.integer, int)):
                # Convert scalar action to a dictionary assigning the same action to all agents
                actions = {agent: int(actions) for agent in self.agents}
            else:
                raise ValueError(f"Invalid type for actions: {type(actions)}. Expected dict or int.")

        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            if self.dones[agent]:
                continue  # Skip if agent is already done

            hand = self.hands[agent]

            # Process action: Discard or draw
            if action < len(hand):  # Discard action
                discarded_card = hand.pop(action)
                self.discard_pile.append(discarded_card)
                rewards[agent] = self._evaluate_hand(hand)
                self._affect_opponents(discarded_card)
            elif action == len(hand):  # Draw action
                if self.deck:
                    hand.append(self.deck.pop())
                else:
                    infos[agent]["error"] = "Deck is empty"

            # Check termination and truncation conditions
            terminated[agent] = self._check_win(hand)
            truncated[agent] = self.rounds >= self.max_rounds

        # Increment round counter
        self.rounds += 1

        # Check if all agents are done
        all_done = all(terminated.values()) or all(truncated.values())
        for agent in self.agents:
            self.dones[agent] = all_done

        # Observations for all agents
        observations = self.observe_all()

        return observations, rewards, terminated, truncated, infos





    
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from pettingzoo import MultiAgentRummyEnv
import optuna
import numpy as np

# Function to initialize the multi-agent environment
def make_env():
    return MultiAgentRummyEnv(max_rounds=100, num_players=2)

# Function for training the multi-agent model
def train_multiagent_model(env_class, total_timesteps=5000, num_envs=2, log_dir="./ppo_rummy_logs/"):
    # Use SubprocVecEnv for parallel environments or DummyVecEnv for single-threaded training
    if num_envs > 1:
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        vec_env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{log_dir}/rummy_ai_model")

    # Close the environment
    vec_env.close()

    return model

# Function for optimizing hyperparameters
def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int('num_envs', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)

    # Wrap environment using DummyVecEnv
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        verbose=0
    )

    model.learn(total_timesteps=5000)
    win_rate = evaluate_model(model)

    env.close()  # Ensure environment is closed properly
    return win_rate

# Function for evaluating the trained model
def evaluate_model(model, num_games=50):
    env = DummyVecEnv([make_env])
    wins = 0

    for _ in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            actions, _ = model.predict(obs)
            obs, rewards, dones, _ = env.step(actions)
            for reward in rewards.values():
                if reward > 0:  # Positive reward indicates a win
                    wins += 1
                    break

    env.close()
    win_rate = wins / num_games
    print(f"Win rate: {win_rate:.2f}")
    return win_rate

# Main execution and workflow
if __name__ == "__main__":
    # Optimize hyperparameters
    study = optuna.create_study(direction="maximize", study_name="Rummy Reward Optimization")
    study.optimize(optimize_hyperparameters, n_trials=10, n_jobs=1)  # Single-threaded

    print("Best parameters:", study.best_params)

    # Train the model with optimized hyperparameters
    model = train_multiagent_model(
        env_class=make_env,
        total_timesteps=100000,  # Total training timesteps
        num_envs=4,             # Number of parallel environments
        log_dir="./ppo_rummy_logs"  # Log directory
    )

    # Evaluate the trained model
    win_rate = evaluate_model(model, num_games=100)
    print(f"Final win rate: {win_rate:.2f}")





from pettingzoo.utils.wrappers import ParallelEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

def optimize_hyperparameters(trial):
    num_envs = trial.suggest_int('num_envs', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.99)

    # Use ParallelEnvWrapper
    env = DummyVecEnv([lambda: ParallelEnvWrapper(MultiAgentRummyEnv(max_rounds=100, num_players=2))])
    
    model = PPO(
        "MlpPolicy",
        env,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        verbose=0
    )

    model.learn(total_timesteps=5000)
    win_rate = evaluate_model(model)

    env.close()  # Ensure environment is closed properly

    return win_rate

def evaluate_model(model, num_games=50):
    env = DummyVecEnv([lambda: ParallelEnvWrapper(MultiAgentRummyEnv(max_rounds=100, num_players=2))])
    wins = 0

    for _ in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            actions, _ = model.predict(obs)
            obs, rewards, dones, _ = env.step(actions)
            for reward in rewards.values():
                if reward > 0:  # Positive reward indicates a win
                    wins += 1
                    break

    env.close()
    win_rate = wins / num_games
    print(f"Win rate: {win_rate:.2f}")
    return win_rate



#Module 5: Benchmark Players

class RandomPlayer:
    """
    Baseline player that chooses random actions.
    """

    def choose_action(self, observation):
        return random.choice(range(54))


class HeuristicPlayer:
    """
    Baseline player with a simple heuristic strategy.
    """

    def choose_action(self, observation):
        # Example heuristic: Discard the highest-value card
        hand = observation[:54]
        high_card_index = np.argmax(hand)
        return high_card_index

#Module 6: Stress Testing

def test_edge_cases():
    """
    Test the AI under challenging scenarios.
    """
    env = MultiAgentRummyEnv(max_rounds=100, num_players=2)

    # Example edge case: Difficult starting hands
    env.hands["player_0"] = [(13, '♥'), (12, '♦'), (11, '♣')]  # Difficult hand
    env.hands["player_1"] = [(1, '♥'), (1, '♦'), (1, '♣')]  # Perfect start

    print("Testing edge cases...")
    obs = env.observe_all()
    actions = {"player_0": 0, "player_1": 0}  # Dummy actions
    obs, rewards, dones, _ = env.step(actions)

    print("Edge case results:", rewards)

#Module 7: Explainability

def explain_decision(observation):
    """
    Explain the AI's decision-making process.
    """
    hand = observation[:54]
    discard_pile = observation[56:110]
    deck_ratio = observation[54]
    round_ratio = observation[55]

    explanation = f"""
    Decision Analysis:
    - Hand: {hand}
    - Discard Pile (recent): {discard_pile}
    - Deck Ratio: {deck_ratio:.2f}
    - Round Progress: {round_ratio:.2f}
    """
    return explanation

#Module 8: Live Dashboard and Visualization

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class LivePlot:
    """
    Simple live plot for training rewards.
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.reward_history = []
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Live Training Reward Plot")
        plt.ion()  # Enable interactive mode

    def update(self, reward):
        """
        Update the live plot with the latest reward.
        """
        self.reward_history.append(np.mean(reward))
        self.ax.clear()
        self.ax.plot(self.reward_history)
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Average Reward")
        self.ax.set_title("Live Training Progress")
        plt.draw()
        plt.pause(0.01)

    def show_plot(self):
        """
        Show the final plot.
        """
        plt.show()


class LiveDashboard:
    """
    Comprehensive live dashboard with multiple metrics.
    """

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.ion()

        # Subplots for different metrics
        self.reward_ax = self.axes[0, 0]
        self.reward_ax.set_title("Reward Over Timesteps")
        self.reward_ax.set_xlabel("Timesteps")
        self.reward_ax.set_ylabel("Rewards")
        self.reward_line, = self.reward_ax.plot([], [], label="Rewards")
        self.avg_reward_line, = self.reward_ax.plot([], [], label="Avg Reward")
        self.reward_ax.legend()

        self.win_loss_ax = self.axes[0, 1]
        self.win_loss_ax.set_title("Win/Loss Count")
        self.win_loss_ax.set_xlabel("Episodes")
        self.win_loss_ax.set_ylabel("Count")
        self.win_line, = self.win_loss_ax.plot([], [], label="Wins")
        self.loss_line, = self.win_loss_ax.plot([], [], label="Losses")
        self.win_loss_ax.legend()

        self.action_ax = self.axes[1, 0]
        self.action_ax.set_title("Action Frequency")
        self.action_ax.set_xlabel("Actions")
        self.action_ax.set_ylabel("Frequency")
        self.action_bars = None

        self.metrics_ax = self.axes[1, 1]
        self.metrics_ax.set_title("Game Metrics")
        self.metrics_ax.set_xlabel("Timesteps")
        self.metrics_ax.set_ylabel("Count")
        self.pure_seq_line, = self.metrics_ax.plot([], [], label="Pure Sequences")
        self.valid_set_line, = self.metrics_ax.plot([], [], label="Valid Sets")
        self.metrics_ax.legend()

        # Data tracking
        self.timesteps = []
        self.rewards = []
        self.avg_rewards = []
        self.wins = []
        self.losses = []
        self.action_counts = Counter()
        self.pure_sequences = []
        self.valid_sets = []

    def update(self, timestep, reward, avg_reward, win_count, loss_count, metrics, action):
        """
        Update the live dashboard with new data.
        """
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.wins.append(win_count)
        self.losses.append(loss_count)
        self.action_counts[action] += 1
        self.pure_sequences.append(metrics.get("pure_sequences", 0))
        self.valid_sets.append(metrics.get("valid_sets", 0))

        # Update reward plot
        self.reward_line.set_data(self.timesteps, self.rewards)
        self.avg_reward_line.set_data(self.timesteps, self.avg_rewards)
        self.reward_ax.relim()
        self.reward_ax.autoscale_view()

        # Update win/loss plot
        self.win_line.set_data(range(len(self.wins)), self.wins)
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.win_loss_ax.relim()
        self.win_loss_ax.autoscale_view()

        # Update action frequency
        if self.action_bars:
            for bar, count in zip(self.action_bars, self.action_counts.values()):
                bar.set_height(count)
        else:
            self.action_bars = self.action_ax.bar(self.action_counts.keys(), self.action_counts.values())
        self.action_ax.relim()
        self.action_ax.autoscale_view()

        # Update metrics plot
        self.pure_seq_line.set_data(self.timesteps, self.pure_sequences)
        self.valid_set_line.set_data(self.timesteps, self.valid_sets)
        self.metrics_ax.relim()
        self.metrics_ax.autoscale_view()

        # Redraw plots
        plt.pause(0.01) 
        plt.draw()

#Module 9: Diverse Self-Play Strategies

class DiversePlayer:
    """
    A player capable of dynamically switching between strategies.
    """

    def __init__(self):
        self.strategies = [RandomPlayer(), HeuristicPlayer()]
        self.current_strategy = random.choice(self.strategies)

    def choose_action(self, observation):
        """
        Select an action based on the current strategy.
        """
        return self.current_strategy.choose_action(observation)

    def switch_strategy(self):
        """
        Dynamically switch to a new strategy.
        """
        self.current_strategy = random.choice(self.strategies)
        
#Module 10: Real-World Integration

import cv2

def capture_game_state(frame):
    """
    Capture the game state from a video frame (e.g., using OpenCV).
    """
    # Example: Detect cards and map to internal representation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add logic to detect cards using object detection or template matching
    detected_cards = []
    return detected_cards


def execute_real_world_action(action):
    """
    Translate AI action into real-world execution.
    """
    # Example: Simulate mouse clicks or keyboard inputs for a game
    print(f"Executing action: {action}")


#Module 11: Main Execution and Workflow

from stable_baselines3.common.vec_env import DummyVecEnv
from optuna import create_study
import numpy as np
import multiprocessing as mp
import sys

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)


    # Initialize the Rummy environment
    env = DummyVecEnv([lambda: MultiAgentRummyEnv(max_rounds=100, num_players=2)])


    # Optimize hyperparameters
    study = create_study(direction="maximize", study_name="Rummy Reward Optimization")
    study.optimize(optimize_hyperparameters, n_trials=10, n_jobs=1)  # Single-threaded


    print("Best parameters:", study.best_params)

    # Train the model with optimized hyperparameters
    model = train_multiagent_model(
        env_class=MultiAgentRummyEnv,
        total_timesteps=100000,  # Total training timesteps
        num_envs=4,             # Number of parallel environments
        log_dir="./ppo_rummy_logs"  # Log directory
    )

    # Evaluate the trained model
    win_rate = evaluate_model(model, num_games=100)
    print(f"Final win rate: {win_rate:.2f}")

    # Example real-world action
    example_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for a video frame
    print("Example frame captured. Ready for game state extraction.")

