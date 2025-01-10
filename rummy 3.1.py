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
        self.cards += [Card('J', None) for _ in range(2)]  # Adding printed Jokers
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
        super(RummyEnv, self).__init__()

        self.action_space = spaces.Discrete(108)  # 54 cards + pick/drop actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(108,), dtype=np.float32)  # Updated to 108

        self.deck = None
        self.discard_pile = []
        self.hand_1 = []
        self.hand_2 = []
        self.current_turn = 1
        self.joker_card = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = Deck()
        self.joker_card = self.deck.set_joker()
        self.discard_pile = [self.deck.draw()]
        self.hand_1 = [self.deck.draw() for _ in range(13)]
        self.hand_2 = [self.deck.draw() for _ in range(13)]
        self.current_turn = 1
        return self._get_observation(), {}

    def step(self, action):
        reward = 0
        done = False
        info = {}

        active_hand = self.hand_1 if self.current_turn == 1 else self.hand_2

        if action < len(active_hand):  # Discard card
            card_to_discard = active_hand.pop(action)
            self.discard_pile.append(card_to_discard)
            reward += 1  # Small reward for discarding
        elif action == len(active_hand):  # Pick from deck
            drawn_card = self.deck.draw()
            if drawn_card:
                active_hand.append(drawn_card)
        elif action == len(active_hand) + 1:  # Pick from discard pile
            if self.discard_pile:
                active_hand.append(self.discard_pile.pop())

        # Check if the game is won
        if self._check_win(active_hand):
            done = True
            reward += 100

        # Penalty for invalid moves
        if action >= len(active_hand) + 2:
            reward -= 10

        # Change turn
        self.current_turn = 3 - self.current_turn
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        def encode_hand(hand):
            encoding = np.zeros(54, dtype=np.float32)
            for card in hand:
                index = self._get_card_index(card)
                encoding[index] = 1
            return encoding

        hand_encoding = encode_hand(self.hand_1 if self.current_turn == 1 else self.hand_2)
        discard_encoding = encode_hand(self.discard_pile)
        return np.concatenate([hand_encoding, discard_encoding])  # Observation now matches (108,)

    def _get_card_index(self, card):
        if card.rank == 'J':
            return 52 + (0 if card.isjoker else 1)
        suits = {'♥': 0, '♦': 1, '♣': 2, '♠': 3}
        return (card.rank - 1) * 4 + suits[card.suit]

    def _check_win(self, hand):
        return self._has_pure_sequence(hand) and (
            self._has_valid_combinations(hand) or self._has_impure_sequence(hand)
        )

    def _has_pure_sequence(self, hand):
        sorted_hand = sorted([card for card in hand if not card.isjoker and card.suit],
                             key=lambda c: (c.suit, c.rank))
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

    def _has_impure_sequence(self, hand):
        sorted_hand = sorted(hand, key=lambda c: (c.suit if c.suit else '', c.rank if c.rank != 'J' else float('inf')))
        sequences = 0
        temp_seq = []

        for card in sorted_hand:
            if temp_seq and card.suit == temp_seq[-1].suit:
                if card.rank == temp_seq[-1].rank + 1:
                    temp_seq.append(card)
                elif card.isjoker:
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

    def _has_valid_combinations(self, hand):
        rank_counts = {}
        for card in hand:
            if card.rank not in rank_counts:
                rank_counts[card.rank] = 0
            rank_counts[card.rank] += 1

        sets = sum(1 for count in rank_counts.values() if count >= 3)
        return sets >= 2

if __name__ == "__main__":
    env = DummyVecEnv([lambda: RummyEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    model.save("rummy_model")
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Game Over")
            break
