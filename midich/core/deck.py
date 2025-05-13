from enum import Enum
from .exceptions import InvalidCardValue, EmptyDeckAccess
import itertools
import random


class CardSuit(Enum):
    DIAMOND = 1
    HEARTS = 2
    SPADES = 3
    CLUBS = 4

    def __str__(self) -> str:
        if self.name == "DIAMOND":
            return "♦"
        elif self.name == "HEARTS":
            return "♥"
        elif self.name == "SPADES":
            return "♣"
        return "♠"
    
    def __repr__(self) -> str:
        return str(self)
    

class CardValue(Enum):
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 1

    def __str__(self) -> None:
        if self.value <= 10:
            return str(self.value)
        return {
            11: "J",
            12: "Q",
            13: "K",
            1: "A"
        }[self.value]

    @classmethod
    def from_str(cls, new_value: str) -> "CardValue":
        if new_value in {"6", "7", "8", "9", "10"}:
            return cls(int(new_value))
        mapping = {"J": cls.JACK, "Q": cls.QUEEN, "K": cls.KING, "A": cls.ACE}
        if new_value in mapping:
            return mapping[new_value]
        raise InvalidCardValue(new_value)


class Card:
    def __init__(self, suit: CardSuit, value: CardValue) -> None:
        self.suit = suit
        self.value = value
    
    @property
    def rank(self) -> int:
        return self.value.value
    
    def __str__(self) -> str:
        return str(self.value) + str(self.suit)
    
    def __repr__(self) -> str:
        return str(self)


POSSIBLE_VALUES: list[CardValue] = [
    CardValue.SIX,
    CardValue.SEVEN,
    CardValue.EIGHT,
    CardValue.NINE,
    CardValue.TEN,
    CardValue.JACK,
    CardValue.QUEEN,
    CardValue.KING,
    CardValue.ACE
]


POSSIBLE_SUITS: list[CardSuit] = [
    CardSuit.DIAMOND,
    CardSuit.HEARTS,
    CardSuit.SPADES,
    CardSuit.CLUBS
]


class Deck:
    def __init__(self) -> None:
        self.cards: list[Card] = [Card(suit, value) for suit, value in itertools.product(POSSIBLE_SUITS, POSSIBLE_VALUES)]
    
    def shuffle(self) -> None:
        random.shuffle(self.cards)
    
    def __str__(self) -> str:
        return " ".join([str(card) for card in self.cards])
    
    def __repr__(self) -> str:
        return str(self)
    
    def pop(self) -> Card:
        if len(self.cards) == 0:
            raise EmptyDeckAccess()
        return self.cards.pop()
    
    def __bool__(self) -> bool:
        return len(self.cards) > 0
