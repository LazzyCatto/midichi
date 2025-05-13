from .deck import Card, Deck
from  midich.utils import print_table
from .exceptions import GuidedMissingParameters


def is_similar(lhs: Card, rhs: Card) -> bool:
    return lhs.suit == rhs.suit or lhs.value == rhs.value


def collapse_table(table: list[Card], guided: bool, **kwargs) -> None:
    if guided:
        missing_params: set[str] = {"sleep_time", "suit_colors"} - set(kwargs.keys())
        if missing_params:
            raise GuidedMissingParameters(missing_params)

    current: int = len(table) - 2
    while current < len(table) - 1:
        left = table[current - 1]
        right = table[current + 1]
        if is_similar(left, right):
            if guided:
                print_table(
                    table,
                    sleep_time=kwargs["sleep_time"],
                    suit_colors=kwargs["suit_colors"],
                    highlight_indices=(current - 1, current + 1)
                    )
            del table[current - 1]
            current -= 2
            if current < 1:
                current = 1
        else:
            current += 1


def midich(guided: bool = False, **kwargs) -> list[Card]:
    deck: Deck = Deck()
    deck.shuffle()

    table = [deck.pop() for _ in range(2)]

    if guided:
        missing_params: set[str] = {"sleep_time", "suit_colors", "add_color"} - set(kwargs.keys())
        if missing_params:
            raise GuidedMissingParameters(missing_params)
    
    while deck:
        table = [*table, deck.pop()]
        if guided:
            print_table(
                table,
                sleep_time=kwargs["sleep_time"],
                suit_colors=kwargs["suit_colors"],
                highlight_indices=(),
                add_color=kwargs["add_color"]
                )
            collapse_table(
                table,
                guided=guided,
                sleep_time=kwargs["sleep_time"],
                suit_colors=kwargs["suit_colors"]
                )
        else:
            collapse_table(table, guided=False)
    if guided:
        print_table(
            table,
            sleep_time=kwargs["sleep_time"],
            suit_colors=kwargs["suit_colors"],
            highlight_indices=()
            )

    return table