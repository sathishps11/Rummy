from pettingzoo import AECEnv  # Keep this import as it’s correct
from gymnasium import spaces
import numpy as np
import random
from collections import Counter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
import optuna
import matplotlib.pyplot as plt
import gymnasium as gym


class MultiAgentRummyEnv(gym.Env):
    """
    Multi-agent Rummy environment adhering to Gymnasium standards.
    """
    metadata = {"render_modes": ["human"], "name": "multi_agent_rummy"}

    def __init__(self, max_rounds=100, num_players=2, num_jokers=2):
        super(MultiAgentRummyEnv, self).__init__()
        self.max_rounds = max_rounds
        self.num_players = num_players
        self.num_jokers = num_jokers

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

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new game.
        """
        self.rounds = 0
        self.deck = self._create_deck()
        random.shuffle(self.deck)

        self.hands = {f"player_{i}": [self.deck.pop() for _ in range(13)] for i in range(self.num_players)}
        self.discard_pile = [self.deck.pop()]
        self.rewards = {agent: 0 for agent in self.hands.keys()}
        self.dones = {agent: False for agent in self.hands.keys()}

    # Return observations for all agents
        return self.observe_all()

    
    # Define a card-to-index mapping, assuming a deck of 52 cards (replace with actual card names)
    card_to_index = {
       '2H': 0, '3H': 1, '4H': 2, '5H': 3, '6H': 4, '7H': 5, '8H': 6, '9H': 7, '10H': 8, 'JH': 9, 'QH': 10, 'KH': 11,
       'AH': 12, '2D': 13, '3D': 14, '4D': 15, '5D': 16, '6D': 17, '7D': 18, '8D': 19, '9D': 20, '10D': 21, 'JD': 22, 'QD': 23, 'KD': 24,
       'AD': 25, '2C': 26, '3C': 27, '4C': 28, '5C': 29, '6C': 30, '7C': 31, '8C': 32, '9C': 33, '10C': 34, 'JC': 35, 'QC': 36, 'KC': 37,
       'AC': 38, '2S': 39, '3S': 40, '4S': 41, '5S': 42, '6S': 43, '7S': 44, '8S': 45, '9S': 46, '10S': 47, 'JS': 48, 'QS': 49, 'KS': 50,
       'AS': 51
    }
    def observe_all(self):
        """
        Returns numeric observations for all agents as a NumPy array.
        Each observation includes:
      - Encoded hand (52 slots for cards)
      - Deck ratio (1 slot)
      - Round ratio (1 slot)
      - Discard pile history (54 slots)
        """
        observations = []


        for agent in self.hands.keys():
        # Create a zero array for the hand
            obs = np.zeros(54, dtype=np.float32)

        # Encode hand into the observation vector
            for card in self.hands[agent]:
                obs[self._card_to_index(card)] += 1

        # Add metadata
            deck_ratio = len(self.deck) / 54
            round_ratio = self.rounds / self.max_rounds

        # Discard pile history (last 3 cards)
            discard_obs = np.zeros(54, dtype=np.float32)
            for card in self.discard_pile[-3:]:
                discard_obs[self._card_to_index(card)] += 1

        # Combine all components into a single observation
            obs = np.concatenate([obs, [deck_ratio, round_ratio], discard_obs])

        # Append the observation to the list
            observations.append(obs)

    # Return as a stacked NumPy array
        return np.array(observations, dtype=np.float32)







    def observe(self, agent):
       """
       Generate an observation for a specific agent.
       """
       hand = self.hands[agent]
       obs = np.zeros(54, dtype=np.float32)  # Ensure correct size (54 slots)

    # Encode the player's hand
       for card in hand:
           card_index = self._card_to_index(card)
           if card_index == -1:
              print(f"Warning: Invalid card {card}, skipping.")
              continue  # Skip invalid cards
  
       if card_index >= 0 and card_index < 54:  # Ensure the index is within valid bounds (0 to 53)
          obs[card_index] += 1

    # Add discard pile and game metadata
       discard_obs = np.zeros(54, dtype=np.float32)
       for card in self.discard_pile[-3:]:
           discard_index = self._card_to_index(card)
           if discard_index < 54:  # Ensure the index is within valid bounds (0 to 53)
              discard_obs[discard_index] += 1

       deck_ratio = len(self.deck) / 54
       round_ratio = self.rounds / self.max_rounds

       obs = np.concatenate([obs, [deck_ratio, round_ratio], discard_obs])
       obs = np.pad(obs, (0, 168 - obs.size), constant_values=0)  # Ensure the final observation is 168-length
       return obs

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            if self.dones[agent]:
               continue

            hand = self.hands[agent]

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
                  self.dones[agent] = True  # End game for the agent
                  continue

            self.dones[agent] = self._check_win(hand)

        self.rounds += 1
        if self.rounds >= self.max_rounds or all(self.dones.values()):
           for agent in self.agents:
               self.truncateds[agent] = True

        observations = self.observe_all()
        return observations, rewards, self.dones, self.truncateds, infos


    def render(self, mode="human"):
        print(f"Round: {self.rounds}, Discard Pile: {self.discard_pile}")

    def _create_deck(self):
        suits = ["♥", "♦", "♣", "♠"]
        ranks = list(range(1, 14))
        return [(rank, suit) for suit in suits for rank in ranks] + ["joker"] * self.num_jokers

    def _card_to_index(self, card):
        """
        Convert a card to an index for the observation vector.
        """
        if card == 'joker':
           return 52  # Jokers are indexed at 52
        rank, suit = card
        suit_offset = {'♥': 0, '♦': 13, '♣': 26, '♠': 39}.get(suit, -1)
        if suit_offset == -1:
           raise ValueError(f"Invalid suit {suit}")
        return suit_offset + rank - 1



    def _evaluate_hand(self, hand):
        """
        Evaluate the current player's hand with strategic rewards.
        """
        reward = 0
        has_pure_sequence = self._has_pure_sequence(hand)
        has_valid_combinations = self._has_valid_combinations(hand)

        if has_pure_sequence:
            reward += 50
        if has_valid_combinations:
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

# Gym wrapper class to make it compatible with Stable Baselines3
class GymWrapper(gym.Env):
    def __init__(self, env: MultiAgentRummyEnv):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.agents = env.agents

    def reset(self):
    # Example logic for resetting the environment, update according to your actual logic.
       player_index = 0  # Assuming this is player 0, replace with logic to get the actual player index.
    
    # Example cards as tuples of (value, suit), replace with your actual card-hand logic
       cards_player_0 = [(10, '♥'), (8, '♣'), (13, '♥'), (2, '♣')]  
       cards_player_1 = [(6, '♣'), (2, '♥'), (1, '♦'), (4, '♦')]  
    
    # You might need to map suits and values to numerical representations
       def card_to_number(card):
           value_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
           suit_map = {'♠': 1, '♣': 2, '♦': 3, '♥': 4}  # Example suit mapping
           value, suit = card
           return value_map[value], suit_map[suit]
    
    # Convert card hands to numerical format
       num_cards_player_0 = [card_to_number(card) for card in cards_player_0]
       num_cards_player_1 = [card_to_number(card) for card in cards_player_1]
    
    # Flatten the card hands and create the observation
       obs = [player_index] + [num for card in num_cards_player_0 + num_cards_player_1 for num in card]
    
       print(f"Observing at reset: {obs}")  # This will print the observation in numerical format
       return obs



    def step(self, action):
    # Your existing step logic
       obs, reward, done, info = self._get_step_observation(action)  # Replace with your actual logic
       print(f"Observing after action {action}: {obs}")  # This will print the observation after each action is taken
       return obs, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)


# Helper Functions

def make_env():
    return MultiAgentRummyEnv(max_rounds=100, num_players=2)

def train_multiagent_model(env_class, total_timesteps=5000, num_envs=2, log_dir="./ppo_rummy_logs/"):
    if num_envs > 1:
        vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        vec_env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, batch_size=128, gamma=0.99)

    # Initialize LiveDashboard
    dashboard = LiveDashboard()
    timesteps = []
    rewards = []
    avg_rewards = []
    win_counts = []
    loss_counts = []

    # Training loop
    for step in range(total_timesteps):
        # Simulate data collection (replace with real training metrics)
        timestep = step
        reward = np.random.random() * 100  # Replace with actual reward
        avg_reward = np.mean(rewards[-10:]) if rewards else reward
        win_count = step // 100  # Replace with actual win count
        loss_count = (step // 100) // 2  # Replace with actual loss count

        timesteps.append(timestep)
        rewards.append(reward)
        avg_rewards.append(avg_reward)
        win_counts.append(win_count)
        loss_counts.append(loss_count)

        # Update dashboard
        dashboard.update(timestep, reward, avg_reward, win_count, loss_count)

    # Finalize training
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{log_dir}/rummy_ai_model")
    vec_env.close()

    # Display final dashboard
    plt.show()

    return model


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
                if reward > 0:
                    wins += 1
                    break

    env.close()
    win_rate = wins / num_games
    print(f"Win rate: {win_rate:.2f}")
    return win_rate






if __name__ == "__main__":
    # Test the environment setup
    print("Initializing the environment...")
    env = MultiAgentRummyEnv(max_rounds=100, num_players=2)
    obs = env.reset()
    print(f"Initial observations: {obs}")
    print(f"Observation type: {type(obs)}, dtype: {obs.dtype if isinstance(obs, np.ndarray) else 'N/A'}, shape: {obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")

    # Start the Optuna study
    try:
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize", study_name="Rummy Reward Optimization")
        study.optimize(optimize_hyperparameters, n_trials=10, n_jobs=1)
        print("Best parameters:", study.best_params)
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
        exit(1)

    # Train the final model
    try:
        print("Training the final model with optimized parameters...")
        model = train_multiagent_model(
            env_class=make_env,
            total_timesteps=100000,
            num_envs=4,
            log_dir="./ppo_rummy_logs"
        )
    except Exception as e:
        print(f"Error during model training: {e}")
        exit(1)

    # Evaluate the model
    try:
        print("Evaluating the final model...")
        win_rate = evaluate_model(model, num_games=100)
        print(f"Final win rate: {win_rate:.2f}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")



from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Callback to log rewards and visualize training progress.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reward_history = []

    def _on_step(self):
        rewards = self.locals.get("rewards", [])
        if rewards:
            self.reward_history.append(np.mean(rewards))
        return True

    def on_training_end(self):
        # Plot reward history if needed
        plt.plot(self.reward_history)
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.title("Reward Progress Over Time")
        plt.show()

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
        return np.argmax(hand)

def test_edge_cases():
    cases = [
        {"player_0": [(13, '♥'), (12, '♦'), (11, '♣')], "player_1": [(1, '♥'), (1, '♦'), (1, '♣')]},
        {"player_0": ["joker"] * 13, "player_1": [(1, '♥'), (1, '♦'), (1, '♣')]},
    ]
    env = MultiAgentRummyEnv(max_rounds=100, num_players=2)

    for case in cases:
        env.hands.update(case)
        print("Testing case:", case)
        obs = env.observe_all()
        actions = {agent: 0 for agent in env.agents}
        obs, rewards, dones, infos = env.step(actions)
        print("Results:", rewards)


def explain_decision(model, observation):
    action_probs, _ = model.predict(observation, deterministic=False)
    explanation = f"Action Probabilities: {action_probs}"
    return explanation


import cv2


def capture_game_state(frame):
    """
    Capture the game state from a video frame (e.g., using OpenCV).
    """
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

    def update(self, timestep, reward, avg_reward, win_count, loss_count):
        """
        Update the live dashboard with new data.
        """
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.wins.append(win_count)
        self.losses.append(loss_count)

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

        # Redraw plots
        plt.pause(0.01)
        plt.draw()

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

def test_special_cases():
    """
    Test the AI under specific and extreme conditions.
    """
    env = MultiAgentRummyEnv(max_rounds=100, num_players=2)

    # Case 1: All Jokers in a Hand
    env.hands["player_0"] = ["joker"] * 13
    env.hands["player_1"] = [(1, '♥'), (1, '♦'), (1, '♣')]  # Valid set

    print("Testing special cases...")
    obs = env.observe_all()
    actions = {"player_0": 0, "player_1": 0}  # Dummy actions
    obs, rewards, dones, infos = env.step(actions)

    print("Special case results:", rewards)

from stable_baselines3.common.env_util import make_vec_env


def train_distributed_model(env_class, total_timesteps=100000, num_envs=8, log_dir="./ppo_rummy_logs/"):

    """
    Train the model in a distributed setup using multiple environments.
    """

    vec_env = make_vec_env(env_class, n_envs=num_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=128,
        gamma=0.99,
        n_steps=2048
    )
    callback = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(f"{log_dir}/rummy_ai_distributed_model")

    vec_env.close()
    return model

def simulate_real_game(model, num_players=2):
    """
    Simulate a real-world game scenario to evaluate the trained model.
    """
    env = MultiAgentRummyEnv(max_rounds=100, num_players=num_players)
    obs = env.reset()
    done = False

    print("Starting a real-world simulation...")
    while not done:
        actions = {}
        for agent in env.agents:
            actions[agent], _ = model.predict(obs[agent])
        obs, rewards, dones, infos = env.step(actions)

        if all(dones.values()):
            done = True

    print(f"Game Over. Rewards: {rewards}")
    env.render()

def test_unusual_conditions():
    """
    Test the AI's ability to handle non-standard rules or unusual conditions.
    """
    env = MultiAgentRummyEnv(max_rounds=50, num_players=2)

    # Example: Custom rule - No Jokers
    env.deck = [(rank, suit) for suit in ["♥", "♦", "♣", "♠"] for rank in range(1, 14)]  # No Jokers
    random.shuffle(env.deck)

    obs = env.reset()
    done = False
    print("Testing under custom conditions...")

    while not done:
        actions = {agent: random.randint(0, 53) for agent in env.agents}
        obs, rewards, dones, infos = env.step(actions)

        if all(dones.values()):
            done = True

    print("Custom condition test complete.")
    env.render()

def visualize_game_progress(env):
    """
    Visualize the progression of the game using graphical overlays or summaries.
    """
    rounds = env.rounds
    discard_pile = env.discard_pile
    print(f"Round: {rounds}")
    print(f"Discard Pile: {discard_pile}")
    for agent in env.agents:
        print(f"{agent}'s Hand: {env.hands[agent]}")
