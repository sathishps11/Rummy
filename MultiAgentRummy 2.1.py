from pettingzoo.utils import AECEnv
import gym
from gymnasium import spaces
import numpy as np
from MultiAgentRummy import MultiAgentRummyEnv 
import matplotlib.pyplot as plt


class CustomPettingZooWrapper(gym.Env):
    metadata = {"render_modes": ["human"], "name": "multi_agent_rummy"}

    def __init__(self, num_players=2, max_rounds=100, rules=None):
        super().__init__()
        self.num_players = num_players
        self.max_rounds = max_rounds
        self.rounds = 0
        self.rules = rules or {"use_jokers": True}
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]
        self.agent_selection = self.agents[0]
        self.rewards = {agent: 0 for agent in self.agents}  # Initialize rewards

        # Define observation and action spaces
        self.action_space = spaces.Discrete(54)  # Actions: draw/discard
        self.observation_space = spaces.Box(low=0, high=1, shape=(168,), dtype=np.float32)

        # Initialize game state
        self.deck = self._create_deck()
        if not self.rules["use_jokers"]:
            self.deck = [(rank, suit) for suit in ["\u2665", "\u2666", "\u2663", "\u2660"] for rank in range(1, 14)]
        self.hands = {agent: [] for agent in self.agents}
        self.discard_pile = []

    def reset(self, seed=None, options=None):
        self.rounds = 0
        self.deck = self._create_deck()
        np.random.shuffle(self.deck)
        self.hands = {agent: [self.deck.pop() for _ in range(13)] for agent in self.agents}
        self.discard_pile = [self.deck.pop()]
        self.agent_selection = self.agents[0]
        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection
        if action < len(self.hands[agent]):  # Discard action
            discarded_card = self.hands[agent].pop(action)
            self.discard_pile.append(discarded_card)
        elif action == len(self.hands[agent]) and self.deck:  # Draw action
            self.hands[agent].append(self.deck.pop())
        else:
            raise ValueError("Invalid action!")

        # Check for win conditions and assign rewards
        self._check_win_conditions(agent)

        # Move to the next agent
        self.rounds += 1
        self._advance_to_next_agent()

    def _check_win_conditions(self, agent):
        # Check for winning condition placeholder
        reward = 0  # Placeholder for reward logic
        self.rewards[agent] = reward

    def observe(self, agent):
        hand_obs = np.zeros(54, dtype=np.float32)
        for card in self.hands[agent]:
            hand_obs[self._card_to_index(card)] += 1

        deck_ratio = len(self.deck) / 54
        round_ratio = self.rounds / self.max_rounds

        discard_obs = np.zeros(54, dtype=np.float32)
        for card in self.discard_pile[-3:]:
            discard_obs[self._card_to_index(card)] += 1

        return np.concatenate([hand_obs, [deck_ratio, round_ratio], discard_obs])

    def _create_deck(self):
        suits = ["\u2665", "\u2666", "\u2663", "\u2660"]
        ranks = range(1, 14)
        return [(rank, suit) for suit in suits for rank in ranks] + ["joker"] * 2

    def _card_to_index(self, card):
        if card == "joker":
            return 52
        rank, suit = card
        suit_offset = {"\u2665": 0, "\u2666": 13, "\u2663": 26, "\u2660": 39}[suit]
        return suit_offset + rank - 1

    def _advance_to_next_agent(self):
        idx = self.agents.index(self.agent_selection)
        self.agent_selection = self.agents[(idx + 1) % self.num_players]


from collections import Counter

class MultiAgentRummyEnv(AECEnv):
    # Adding reward logic to the base environment
    
     def _check_win_conditions(self, agent):
        """
        Evaluate the agent's hand for win conditions.
        Assign rewards for valid sequences and sets.
        Penalize for invalid or incomplete hands.
        """
        hand = self.hands[agent]
        reward = 0

    # Check for pure sequences
        if self._has_pure_sequence(hand):
           reward += 50  # Reward for a pure sequence

    # Check for valid combinations (sets or sequences)
        if self._has_valid_combinations(hand):
           reward += 30  # Reward for valid combinations

    # Penalty for not progressing toward a valid hand
        if not self._has_pure_sequence(hand) and not self._has_valid_combinations(hand):
           reward -= 10

    # Game-winning condition
        if self._has_pure_sequence(hand) and self._has_valid_combinations(hand):
           self.dones[agent] = True  # Mark as game won
           reward += 100  # Bonus for winning the game

    # Assign reward
        self.rewards[agent] = reward


def _has_pure_sequence(self, hand):
    """
    Check if the hand contains at least one pure sequence (3+ consecutive cards of the same suit).
    """
    # Filter out jokers and sort the hand by suit and rank
    sorted_hand = sorted([card for card in hand if card != "joker"], key=lambda x: (x[1], x[0]))
    
    temp_seq = []

    for card in sorted_hand:
        # Check if temp_seq is not empty and the card belongs to the same suit as the last card in temp_seq
        if temp_seq and card[1] == temp_seq[-1][1] and card[0] == temp_seq[-1][0] + 1:
            temp_seq.append(card)
        else:
            # If sequence is broken, check if the previous sequence was valid
            if len(temp_seq) >= 3:
                return True
            temp_seq = [card]  # Start a new sequence with the current card

    # Final check for the last sequence
    return len(temp_seq) >= 3


    def _has_valid_combinations(self, hand):
        """
        Check if the hand contains at least two valid sets or sequences.
        """
        rank_counts = Counter(card[0] for card in hand if card != "joker")
        sets = sum(1 for count in rank_counts.values() if count >= 3)

        sequences = 0
        sorted_hand = sorted([card for card in hand if card != "joker"], key=lambda x: (x[1], x[0]))
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


def observe(self, agent):
    """
    Generate an enhanced observation for the given agent.
    Includes:
    - Encoded hand
    - Deck ratio
    - Round ratio
    - Discard pile history
    - Opponent hand sizes
    - Agent-specific metrics
    """
    # Encode the agent's hand
    hand_obs = np.zeros(54, dtype=np.float32)
    for card in self.hands[agent]:
        hand_obs[self._card_to_index(card)] += 1

    # Deck and round ratios
    deck_ratio = len(self.deck) / 54
    round_ratio = self.rounds / self.max_rounds

    # Encode recent discard pile
    discard_obs = np.zeros(54, dtype=np.float32)
    for card in self.discard_pile[-3:]:
        discard_obs[self._card_to_index(card)] += 1

    # Opponent hand sizes
    opponent_hand_sizes = [len(self.hands[opponent]) for opponent in self.agents if opponent != agent]

    # Calculate agent-specific metrics
    hand_strength = sum(card[0] for card in self.hands[agent] if card != "joker")

    # Combine all observations
    return np.concatenate([hand_obs, [deck_ratio, round_ratio], discard_obs, opponent_hand_sizes, [hand_strength]])

class RuleBasedPlayer:
    """
    Baseline player that prioritizes creating sequences and sets.
    """
    def choose_action(self, observation):
        # Example: Discard cards least likely to form a sequence or set
        hand = observation[:54]
        card_values = [(i, value) for i, value in enumerate(hand) if value > 0]
        # Sort cards by likelihood of forming a sequence/set
        sorted_cards = sorted(card_values, key=lambda x: x[1])
        return sorted_cards[0][0]  # Discard the lowest-ranked card
    
import matplotlib.pyplot as plt

class SimplePlotter:
    def __init__(self, title="Training Progress", xlabel="Steps", ylabel="Rewards"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data = []

    def update(self, value):
        self.data.append(value)
        self.plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data, label="Reward Trend")
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()




from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

class MultiAgentTrainer:
    def __init__(self, env_class, num_agents=2, log_dir="./multiagent_logs", total_timesteps=100000):
        """
        Initialize the multi-agent training framework.
        """
        self.env_class = env_class
        self.num_agents = num_agents
        self.total_timesteps = total_timesteps
        self.log_dir = log_dir
        self.models = {}

        # Create a unique environment instance for each agent
        self.envs = [make_vec_env(env_class, n_envs=1) for _ in range(num_agents)]

        # Directory setup for logs and models
        os.makedirs(self.log_dir, exist_ok=True)

    def initialize_agents(self):
        """
        Initialize PPO models for each agent.
        """
        for i in range(self.num_agents):
            agent_log_dir = os.path.join(self.log_dir, f"agent_{i}")
            os.makedirs(agent_log_dir, exist_ok=True)
            model = PPO(
                "MlpPolicy",
                self.envs[i],
                verbose=1,
                tensorboard_log=agent_log_dir,
                batch_size=128,
                gamma=0.99,
            )
            self.models[f"agent_{i}"] = model

    def train_agents(self):
        """
        Train each agent in a self-play setup.
        """
        for i, model in self.models.items():
            print(f"Training {i}...")
            model.learn(total_timesteps=self.total_timesteps)
            model.save(os.path.join(self.log_dir, f"{i}_model"))

    def evaluate_agents(self, num_games=50):
        """
        Evaluate agents by simulating games.
        """
        env = self.env_class()
        wins = {agent: 0 for agent in self.models.keys()}

        for _ in range(num_games):
            obs = env.reset()
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                actions = {}
                for agent in env.agents:
                    if not done[agent]:
                        model = self.models[agent]
                        action, _ = model.predict(obs[env.agents.index(agent)], deterministic=True)
                        actions[agent] = action

                obs, rewards, done, _, _ = env.step(actions)

            for agent, reward in rewards.items():
                if reward > 0:
                    wins[agent] += 1

        # Calculate win rates
        win_rates = {agent: wins[agent] / num_games for agent in wins.keys()}
        print("Win Rates:", win_rates)
        return win_rates

class PerformanceEvaluator:
    def __init__(self, env_class, models, num_games=50):
        """
        Initialize the evaluator.
        :param env_class: The multi-agent environment class.
        :param models: Dictionary of trained agent models.
        :param num_games: Number of games to simulate for evaluation.
        """
        self.env_class = env_class
        self.models = models
        self.num_games = num_games

    def simulate_games(self):
        """
        Simulate games between trained agents and calculate win rates.
        """
        print("=== Evaluating Performance ===")
        env = self.env_class()
        wins = {agent: 0 for agent in self.models.keys()}

        for _ in range(self.num_games):
            obs = env.reset()
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                actions = {}
                for agent in env.agents:
                    if not done[agent]:
                        model = self.models[agent]
                        action, _ = model.predict(obs[env.agents.index(agent)], deterministic=True)
                        actions[agent] = action

                obs, rewards, done, _, _ = env.step(actions)

            for agent, reward in rewards.items():
                if reward > 0:
                    wins[agent] += 1

        # Calculate and print win rates
        win_rates = {agent: wins[agent] / self.num_games for agent in wins.keys()}
        print("Win Rates:", win_rates)
        return win_rates

    def benchmark_against_baseline(self, baseline_player):
        """
        Benchmark trained agents against a baseline player.
        :param baseline_player: An instance of a baseline player (e.g., RandomPlayer, HeuristicPlayer).
        """
        print("=== Benchmarking Against Baseline ===")
        env = self.env_class()
        wins = {agent: 0 for agent in self.models.keys()}
        wins["baseline"] = 0

        for _ in range(self.num_games):
            obs = env.reset()
            done = {agent: False for agent in env.agents}

            while not all(done.values()):
                actions = {}
                for agent in env.agents:
                    if not done[agent]:
                        if agent == "baseline":
                            actions[agent] = baseline_player.choose_action(obs[env.agents.index(agent)])
                        else:
                            model = self.models[agent]
                            action, _ = model.predict(obs[env.agents.index(agent)], deterministic=True)
                            actions[agent] = action

                obs, rewards, done, _, _ = env.step(actions)

            for agent, reward in rewards.items():
                if reward > 0:
                    wins[agent] += 1

        # Calculate and print win rates
        win_rates = {agent: wins[agent] / self.num_games for agent in wins.keys()}
        print("Win Rates (Baseline vs. AI):", win_rates)
        return win_rates


class RandomPlayer:
    """
    Baseline player that chooses random actions.
    """
    def choose_action(self, observation):
        return np.random.randint(0, 54)

class HeuristicPlayer:
    """
    Baseline player with a simple heuristic strategy.
    """
    def choose_action(self, observation):
        # Example heuristic: Discard the highest-value card
        hand = observation[:54]
        return np.argmax(hand)

import pyautogui
import cv2
import numpy as np

class GameStateCapture:
    def __init__(self, path):
        print(f"GameStateCapture initialized with path: {path}")
    def detect_cards(self, frame):
        return []  # Placeholder: No detected cards


    def _load_card_templates(self, path):
        """
        Load card templates for object detection.
        """
        templates = {}
        for card_name in os.listdir(path):
            card_image = cv2.imread(os.path.join(path, card_name), cv2.IMREAD_GRAYSCALE)
            templates[card_name.split('.')[0]] = card_image
        return templates

    def detect_cards(self, frame):
        """
        Detect cards in the given frame using template matching.
        :param frame: Input frame (image) from the game.
        :return: List of detected cards.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_cards = []

        for card_name, template in self.card_templates.items():
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.8)  # Confidence threshold
            for pt in zip(*loc[::-1]):
                detected_cards.append(card_name)
                cv2.rectangle(frame, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

        return detected_cards

class ActionExecutor:
    def __init__(self, mappings):
        print("ActionExecutor initialized.")
    def execute_action(self, action):
        print(f"Executing action: {action}")

    def execute_action(self, action):
        """
        Execute the specified action.
        :param action: Action ID returned by the AI.
        """
        if action in self.action_mappings:
            command = self.action_mappings[action]
            if "click" in command:
                pyautogui.click(command["click"])
            elif "key" in command:
                pyautogui.press(command["key"])

class HumanAITest:
    def __init__(self, ai_model, capture, executor):
        """
        Initialize the human-AI test framework.
        :param ai_model: Trained AI model.
        :param capture: GameStateCapture instance.
        :param executor: ActionExecutor instance.
        """
        self.ai_model = ai_model
        self.capture = capture
        self.executor = executor

    def play_game(self):
        """
        Simulate a game between the AI and human players.
        """
        while True:
            # Capture the current game state
            frame = pyautogui.screenshot()
            frame_np = np.array(frame)
            detected_cards = self.capture.detect_cards(frame_np)

            # Generate AI action
            observation = self._convert_to_observation(detected_cards)
            action, _ = self.ai_model.predict(observation, deterministic=True)

            # Execute the AI's action
            self.executor.execute_action(action)

            # Break condition for testing
            if self._check_game_end():
                print("Game Over!")
                break

    def _convert_to_observation(self, detected_cards):
        observation = np.zeros(54, dtype=np.float32)
        for card in detected_cards:
            index = self.capture._card_to_index(card)
            observation[index] += 1
        return observation

    def _check_game_end(self):
        # Placeholder for game end logic
        return False


from supersuit import PettingZooEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
import os

def train_distributed_model(env_class, num_envs=8, total_timesteps=200_000, log_dir="./distributed_logs"):
    """
    Train the AI model using distributed environments.
    :param env_class: The environment class to be used for training.
    :param num_envs: Number of parallel environments.
    :param total_timesteps: Total training timesteps.
    :param log_dir: Directory for logging and saving models.
    """
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Wrap the environment with PettingZooEnv to make it Gym-compatible
    def make_env():
        pettingzoo_env = env_class()
        return PettingZooEnv(pettingzoo_env)

    # Create multiple parallel environments
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=256,  # Adjust for distributed training
        n_steps=2048,
        gamma=0.99,
    )

    # Train the model
    print("=== Starting Training ===")
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model_path = os.path.join(log_dir, "distributed_rummy_model")
    model.save(model_path)
    vec_env.close()

    print(f"Model saved at {model_path}")
    return model



from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CuriosityRewardCallback(BaseCallback):
    """
    A callback to modify rewards based on curiosity-driven exploration.
    """

    def __init__(self, exploration_weight=0.1, verbose=0):
        super().__init__(verbose)
        self.exploration_weight = exploration_weight
        self.visited_states = set()

    def _on_step(self):
        # Track visited states to reward exploration
        obs = tuple(self.locals["obs"][0].tolist())
        if obs not in self.visited_states:
            self.visited_states.add(obs)
            self.locals["rewards"] += self.exploration_weight  # Add curiosity bonus
        return True

import os
import json
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class EnhancedReplayLoggerCallback(BaseCallback):
    """
    Enhanced callback to log and visualize gameplay data.
    """

    def __init__(self, log_dir="./enhanced_logs", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.replay_data = []
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        # Fetch data from locals safely
        action = self.locals.get("actions", [])
        obs = self.locals.get("obs", [])
        reward = self.locals.get("rewards", [])
        done = self.locals.get("dones", [])

        # Append data ensuring compatibility with JSON
        self.replay_data.append({
            "obs": obs.tolist() if hasattr(obs, "tolist") else obs,
            "action": action.tolist() if hasattr(action, "tolist") else action,
            "reward": float(reward) if isinstance(reward, (int, float)) else reward,
            "done": done.tolist() if hasattr(done, "tolist") else done,
        })

        return True  # Continue training

    def on_training_end(self):
        # Save replay data
        replay_path = os.path.join(self.log_dir, "replay_data.json")
        with open(replay_path, "w") as f:
            json.dump(self.replay_data, f)

        # Plot rewards
        rewards = [step["reward"] for step in self.replay_data if "reward" in step]
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label="Reward per step")
        plt.title("Reward Trends")
        plt.xlabel("Steps")
        plt.ylabel("Rewards")
        plt.legend()
        plot_path = os.path.join(self.log_dir, "reward_trends.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Logs and visualizations saved to {self.log_dir}")
        



import numpy as np

def test_edge_cases(env_class):
    """
    Test the AI under predefined edge cases.
    """
    edge_cases = [
        {
            "name": "All Jokers in Hand",
            "player_0": ["joker"] * 13,
            "player_1": [(1, "♥"), (2, "♦"), (3, "♣")],
        },
        {
            "name": "Difficult Starting Hand",
            "player_0": [(1, "♥"), (4, "♦"), (9, "♣")],
            "player_1": [(10, "♥"), (11, "♦"), (12, "♠")],
        },
        {
            "name": "No Jokers Rule",
            "custom_deck": [(rank, suit) for suit in ["♥", "♦", "♣", "♠"] for rank in range(1, 14)],
        },
    ]

    for case in edge_cases:
        print(f"\nTesting Case: {case['name']}")
        env = env_class()

        # Apply edge case setup
        if "custom_deck" in case:
            env.deck = case["custom_deck"]
        if "player_0" in case and "player_1" in case:
            env.hands["player_0"] = case["player_0"]
            env.hands["player_1"] = case["player_1"]

        obs = env.reset()
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            actions = {agent: np.random.randint(0, env.action_space.n) for agent in env.agents if not done[agent]}
            obs, rewards, done, _, _ = env.step(actions)

        print(f"Rewards: {rewards}")
        env.render()


def test_custom_rules(env_class):
    """
    Test the AI under custom rule variations.
    """
    print("\nTesting with Custom Rules (e.g., No Jokers)...")
    custom_rule_env = env_class(rules={"use_jokers": False})  # No jokers allowed

    obs = custom_rule_env.reset()
    done = {agent: False for agent in custom_rule_env.agents}

    while not all(done.values()):
        actions = {agent: np.random.randint(0, custom_rule_env.action_space.n) for agent in custom_rule_env.agents if not done[agent]}
        obs, rewards, done, _, _ = custom_rule_env.step(actions)

    print(f"Rewards: {rewards}")
    custom_rule_env.render()


def test_adversarial_conditions(env_class, baseline_player):
    """
    Test AI against adversarial scenarios or strong opponents.
    """
    print("\nTesting Adversarial Conditions...")
    env = env_class()
    obs = env.reset()
    done = {agent: False for agent in env.agents}
    rewards = {agent: 0 for agent in env.agents}

    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if not done[agent]:
                if agent == "baseline":
                    actions[agent] = baseline_player.choose_action(obs[env.agents.index(agent)])
                else:
                    actions[agent] = np.random.randint(0, env.action_space.n)  # Replace with trained AI decision

        obs, rewards, done, _, _ = env.step(actions)

    print(f"Rewards: {rewards}")
    env.render()


def run_stress_tests(env_class, baseline_player):
    print("Stress tests are currently disabled. Placeholder function.")
    """
    Run all stress tests: edge cases, custom rules, and adversarial conditions.
    """
    print("=== Running Edge Case Tests ===")
    test_edge_cases(env_class)

    print("\n=== Running Custom Rule Tests ===")
    test_custom_rules(env_class)

    print("\n=== Running Adversarial Tests ===")
    test_adversarial_conditions(env_class, baseline_player)


def main():
    num_agents = 2
    num_envs = 4  # Reduce for debugging
    total_timesteps = 100_000
    log_dir = "./multiagent_logs"
    num_evaluation_games = 5  # Reduce for faster testing
    card_templates_path = "./card_templates"

    print("=== Training AI Models ===")
    model = train_distributed_model(MultiAgentRummyEnv, num_envs=num_envs, total_timesteps=total_timesteps, log_dir=log_dir)

    print("\n=== Evaluating Performance ===")
    evaluator = PerformanceEvaluator(MultiAgentRummyEnv, {"agent_0": model}, num_games=num_evaluation_games)
    win_rates = evaluator.simulate_games()
    print("Win Rates:", win_rates)

    print("\n=== Running Stress Tests ===")
    run_stress_tests(MultiAgentRummyEnv, RandomPlayer())

    print("\n=== Real-World Integration ===")
    capture = GameStateCapture(card_templates_path)
    executor = ActionExecutor({})
    human_ai_test = HumanAITest(model, capture, executor)
    print("Human-AI test game placeholder. Integration skipped.")

if __name__ == "__main__":
    main()

    


