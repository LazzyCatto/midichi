from .deck import Card, Deck
from  midich.utils import print_table


def is_similar(lhs: Card, rhs: Card) -> bool:
    return lhs.suit == rhs.suit or lhs.value == rhs.value


def collapse_table(
        table: list[Card],
        guided: bool,
        sleep_time: float=0.1,
        suit_colors: dict[str, str]={}
        ) -> None:
    current: int = len(table) - 2
    while current < len(table) - 1:
        left = table[current - 1]
        right = table[current + 1]
        if is_similar(left, right):
            if guided:
                print_table(table, sleep_time, suit_colors, (current - 1, current + 1))
            del table[current - 1]
            current -= 2
            if current < 1:
                current = 1
        else:
            current += 1


def midich(
        guided: bool = False,
        sleep_time: float=0.1,
        suit_colors: dict[str, str]={},
        add_color: str="green"
        ) -> list[Card]:
    deck: Deck = Deck()
    deck.shuffle()

    table = [deck.pop() for _ in range(2)]
    
    while deck:
        table = [*table, deck.pop()]
        if guided:
            print_table(table, sleep_time, suit_colors, (), add_color=add_color)
        collapse_table(table, guided=guided, sleep_time=sleep_time, suit_colors=suit_colors)
    if guided:
        print_table(table, sleep_time, suit_colors, ())

    return table