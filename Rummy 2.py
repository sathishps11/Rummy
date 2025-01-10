import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.callbacks import BaseCallback
import torch

# Assuming your RummyEnv is defined correctly in the same script or imported from 'rummy_env.py'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RummyEnv(gym.Env):
    def __init__(self, max_rounds=100):
        super(RummyEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)
        self.action_space = spaces.Discrete(108)
        self.rng = np.random.default_rng()
        self.max_rounds = max_rounds  # Maximum number of rounds before declaring a winner
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.draw_pile = self._create_deck()
        self.rng.shuffle(self.draw_pile)
        self.hand_1 = self._draw_initial_hand()
        self.hand_2 = self._draw_initial_hand()
        self.discard_pile = []
        self.current_turn = 1
        self.rounds = 0  # Track rounds

        observation = self._get_observation()
        return observation, {}

    def _create_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = list(range(1, 14))
        deck = [(rank, suit) for suit in suits for rank in ranks] + ['joker'] * 2
        return deck

    def _draw_initial_hand(self):
        return [self.draw_pile.pop() for _ in range(13)]

    def _get_observation(self):
        obs = [0] * 108
        for card in self.hand_1:
            obs[self._card_to_index(card)] = 1
        for card in self.hand_2:
            obs[self._card_to_index(card)] = 1
        for card in self.discard_pile:
            obs[54 + self._card_to_index(card)] = 1
        return np.array(obs, dtype=np.float32)

    def _card_to_index(self, card):
        if card == 'joker':
            return 52
        rank, suit = card
        suit_offset = {'hearts': 0, 'diamonds': 13, 'clubs': 26, 'spades': 39}[suit]
        return suit_offset + rank - 1

    def step(self, action):
        if action < 54:
            action_type = 0
            card_index = action
        else:
            action_type = 1
            card_index = action - 54

        reward = 0
        terminated = False
        truncated = False
        win = False  # Default win is False

        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        if action_type == 0:
            if 0 <= card_index < len(active_hand):
                discarded_card = active_hand.pop(card_index)
                self.discard_pile.append(discarded_card)
                reward += 1
            else:
                reward -= 5  # Heavier penalty for invalid discard
        elif action_type == 1:
            if len(self.draw_pile) > 0:
                drawn_card = self.draw_pile.pop()
                active_hand.append(drawn_card)
                reward += 0  # Neutral reward for drawing a card
            else:
                terminated = True

        # Check for game-ending conditions
        if len(self.hand_1) == 0:
            terminated = True
            reward += 100  # Player 1 victory
            win = True
        elif len(self.hand_2) == 0:
            terminated = True
            reward += 100  # Player 2 victory
            win = False
        elif self.rounds >= self.max_rounds:  # Max rounds reached, declare winner based on fewer cards
            terminated = True
            win = self._decide_winner_based_on_cards()

        reward += self._evaluate_hand(active_hand)

        self.current_turn = 3 - self.current_turn
        self.rounds += 1  # Increment round count

        observation = self._get_observation()
        info = {'win': win}

        return observation, reward, terminated, truncated, info

    def _evaluate_hand(self, hand):
        # Reward shaping: give bonus for creating sequences
        return 10 * self._count_pure_sequences(hand) + 5 * self._count_impure_sequences(hand)

    def _count_pure_sequences(self, hand):
        return len(hand) // 5  # Simplified for now

    def _count_impure_sequences(self, hand):
        return len(hand) // 10  # Simplified for now

    def _decide_winner_based_on_cards(self):
        # If the max rounds are reached, the player with fewer cards wins
        if len(self.hand_1) < len(self.hand_2):
            return True  # Player 1 wins
        elif len(self.hand_1) > len(self.hand_2):
            return False  # Player 2 wins
        else:
            # In case of equal cards, check sequences
            return self._decide_winner_based_on_sequences()

    def _decide_winner_based_on_sequences(self):
        # Compare the number of sequences each player has formed
        p1_sequences = self._count_pure_sequences(self.hand_1) + self._count_impure_sequences(self.hand_1)
        p2_sequences = self._count_pure_sequences(self.hand_2) + self._count_impure_sequences(self.hand_2)

        if p1_sequences > p2_sequences:
            return True  # Player 1 wins based on more sequences
        elif p2_sequences > p1_sequences:
            return False  # Player 2 wins based on more sequences
        else:
            return None  # If sequences are equal, declare as Draw (should not happen with other checks)

# Reward Logger Callback for collecting rewards over time during training
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps = []  # Store timesteps
        self.rewards = []    # Store rewards

    def _on_step(self) -> bool:
        # Collect reward information and timestep after each step
        self.timesteps.append(self.num_timesteps)  # Current timestep
        self.rewards.append(self.locals['rewards'][0])  # Reward for this timestep
        return True

# TensorBoard Callback (for logging)
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True  # Log at each step

if __name__ == "__main__":
    # Create environment
    env = DummyVecEnv([lambda: RummyEnv()])

    # Specify the directory for TensorBoard logs
    log_dir = "/path/to/log_dir"
    
    # Ensure log_dir exists
    os.makedirs(log_dir, exist_ok=True)

    # Create PPO model with adjusted parameters and TensorBoard logging
    model = PPO(
        'MlpPolicy', env, verbose=1, learning_rate=0.0005, 
        tensorboard_log=log_dir, n_steps=2048, batch_size=64, gamma=0.99,
        ent_coef=0.01  # Correctly set the entropy coefficient for exploration
    )

    # Initialize the reward logger callback
    reward_logger = RewardLoggerCallback()

    # Train the model with the callback
    model.learn(total_timesteps=1000000, callback=reward_logger)

    # Save the trained model
    model.save("rummy_ai_model")

    # Initialize match statistics
    num_matches = 10
    player_1_wins = 0
    player_2_wins = 0
    draws = 0
    action_counts = {}

    # Play matches
    for match in range(num_matches):
        reset_output = env.reset()  # reset returns observation(s) and potentially info
        
        if isinstance(reset_output, tuple):  # If it returns (obs, info)
            obs, info = reset_output
        else:  # If it returns only observation
            obs = reset_output
            info = {}

        done = False
        match_winner = None

        while not done:
            # Model predicts the next action based on observation
            action, _ = model.predict(obs, deterministic=True)

            # Convert the action to an integer if it's a numpy array
            action = action.item() if isinstance(action, np.ndarray) else action

            # Increment the action count
            if action not in action_counts:
                action_counts[action] = 1
            else:
                action_counts[action] += 1

            # Wrap the action in a list to pass it to DummyVecEnv
            action = [action]  # Ensure action is in list format

            try:
                # Step the environment (get new observation, reward, and status)
                step_output = env.step(action)
                
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                else:  # If it returns 4 values
                    obs, reward, terminated, truncated = step_output
                    info = {}

            except ValueError:
                # Fallback if fewer values are returned
                obs, reward, terminated, truncated = env.step(action)
                info = {}

            done = terminated or truncated

            # Track match winner
            if done:
                if info.get('win', None) is not None:
                    match_winner = "Player 1" if info.get('win', False) else "Player 2"
                    if match_winner == "Player 1":
                        player_1_wins += 1
                    else:
                        player_2_wins += 1
                else:
                    match_winner = "Draw"
                    draws += 1

                print(f"Match {match + 1}, Winner: {match_winner}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")

    # Match statistics and action frequencies
    print("\nMatch Summary:")
    print(f"Total Matches: {num_matches}")
    print(f"Player 1 Wins: {player_1_wins}")
    print(f"Player 2 Wins: {player_2_wins}")
    print(f"Draws: {draws}")

    # Print most frequent actions
    print("\nMost Frequent Actions:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"Action {action}: {count} times")

    # Visualize rewards over time (example plot)
    plt.plot(reward_logger.timesteps, reward_logger.rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.savefig("rewards_over_time.png")  # Save the plot as an image
    plt.show()

    # Action frequency heatmap (you should collect actual data)
    action_frequencies = np.zeros(108)  # 108 possible actions (based on your action space)

    # Populate action frequencies based on action_counts data
    for action, count in action_counts.items():
        action_frequencies[action] = count

    # Now let's create a simple heatmap for action frequencies
    # Reshaping the action frequencies for visualization (12x9 grid for display)
    sns.heatmap(action_frequencies.reshape(12, 9), annot=True, fmt='.2f', cmap="YlGnBu")  # Adjusted for floats
    plt.title("Agent's Action Frequencies")
    plt.xlabel("Action Index")
    plt.ylabel("Frequency")
    plt.savefig("action_frequencies_heatmap.png")  # Save heatmap as image
    plt.show()
