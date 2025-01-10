import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.isjoker = False

    def __str__(self):
        if self.isjoker:
            return f"{self.rank}-J"
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        suits = ['♥', '♦', '♣', '♠']  # Hearts, Diamonds, Clubs, Spades
        ranks = list(range(1, 14))
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]
        self.cards += [Card('J', None) for _ in range(2)]
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None

    def set_joker(self):
        joker_card = random.choice(self.cards)
        for card in self.cards:
            if card.rank == joker_card.rank:
                card.isjoker = True
        return joker_card

class RummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(108)  # 54 for cards, pick/drop actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)

        self.deck = None
        self.discard_pile = []
        self.hand = []
        self.joker_card = None
        self.reset()

    def reset(self, seed=None, options=None):
        self.deck = Deck()
        self.joker_card = self.deck.set_joker()
        self.discard_pile = [self.deck.draw()]
        self.hand = [self.deck.draw() for _ in range(13)]
        return self._get_observation(), {}

    def step(self, action):
        reward = 0
        done = False
        truncated = False  # Add truncated variable for compliance

        if action < len(self.hand):  # Discard
            discarded = self.hand.pop(action)
            self.discard_pile.append(discarded)
            reward += 1  # Reward for valid discard
        elif action == len(self.hand):  # Draw from deck
            drawn = self.deck.draw()
            if drawn:
                self.hand.append(drawn)
        elif action == len(self.hand) + 1:  # Draw from discard
            if self.discard_pile:
                self.hand.append(self.discard_pile.pop())

        reward += self._evaluate_hand()

        if self._check_win():
            done = True
            reward += 100  # Large reward for winning

        observation = self._get_observation()
        return observation, reward, done, truncated, {}  # Return all five values

    def _get_observation(self):
        hand_enc = np.zeros(54, dtype=np.float32)
        for card in self.hand:
            idx = self._card_to_index(card)
            hand_enc[idx] = 1

        discard_enc = np.zeros(54, dtype=np.float32)
        for card in self.discard_pile:
            idx = self._card_to_index(card)
            discard_enc[idx] = 1

        return np.concatenate([hand_enc, discard_enc])

    def _card_to_index(self, card):
        suits = {'♥': 0, '♦': 1, '♣': 2, '♠': 3}
        if card.rank == 'J':
            return 52 + (0 if card.isjoker else 1)
        return (card.rank - 1) * 4 + suits[card.suit]

    def _check_win(self):
        # Ensure at least one pure sequence
        if not self._has_pure_sequence():
            return False
        # Ensure valid combinations for the rest of the hand
        if not self._has_valid_combinations():
            return False
        return True

def _has_pure_sequence(self):
    sorted_hand = sorted(
        [card for card in self.hand if not card.isjoker and card.suit],  # Exclude jokers and ensure suit exists
        key=lambda c: (c.suit, c.rank)
    )
    sequences = 0
    temp_seq = []

    for card in sorted_hand:
        if temp_seq and card.suit == temp_seq[-1].suit and card.rank == temp_seq[-1].rank + 1:
            temp_seq.append(card)
        else:
            if len(temp_seq) >= 3:
                sequences += 1
            temp_seq = [card]
    if len(temp_seq) >= 3:
        sequences += 1
    return sequences > 0

def _has_impure_sequence(self):
    sorted_hand = sorted(
        self.hand,
        key=lambda c: (c.suit if c.suit else '', c.rank if c.rank != 'J' else float('inf'))  # Handle None values
    )
    sequences = 0
    temp_seq = []

    for card in sorted_hand:
        if temp_seq and card.suit == (temp_seq[-1].suit if temp_seq[-1].suit else card.suit):
            if card.rank == (temp_seq[-1].rank + 1 if temp_seq else card.rank) or card.isjoker:
                temp_seq.append(card)
            else:
                if len(temp_seq) >= 3:
                    sequences += 1
                temp_seq = [card]
        else:
            if len(temp_seq) >= 3:
                sequences += 1
            temp_seq = [card]
    if len(temp_seq) >= 3:
        sequences += 1
    return sequences > 0


    def _has_valid_combinations(self):
        rank_counts = {}
        for card in self.hand:
            if card.rank not in rank_counts:
                rank_counts[card.rank] = 0
            rank_counts[card.rank] += 1

        sets = sum(1 for count in rank_counts.values() if count >= 3)
        return sets >= 2

    def _evaluate_hand(self):
        reward = 0
        reward += 10 * self._has_pure_sequence()
        reward += 5 * self._has_impure_sequence()
        reward += 5 * self._has_valid_combinations()
        return reward

if __name__ == "__main__":
    env = DummyVecEnv([lambda: RummyEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    model.save("rummy_model")
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
        if done:
            print("Game Over")
            break
