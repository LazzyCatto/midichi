from termcolor import colored
from midich.core.deck import Card
from time import sleep


def get_color(card: Card, suit_colors: dict[str, str]) -> str:
    return suit_colors[card.suit.name]


def colored_card(card: Card, suit_colors: dict[str, str], attrs) -> str:
    return colored(str(card), get_color(card, suit_colors), attrs=attrs)


def print_table(
        cards: list[Card],
        sleep_time: float,
        suit_colors: dict[str, str],
        highlight_indices: tuple[int]=(),
        **kwargs
        ) -> None:
    output = []
    for i, card in enumerate(cards):
        attrs = ["underline"] if i in highlight_indices else []
        if i == len(cards) - 1 and "add_color" in kwargs:
            output.append(colored(str(card), kwargs["add_color"], attrs=attrs))
        else:
            output.append(colored_card(card, suit_colors, attrs=attrs))
    max_length = 111
    current_length = len(" ".join([str(card) for card in cards]))
    print(" ".join(output) + " " * (max_length - current_length), end='\r')
    sleep(sleep_time)
