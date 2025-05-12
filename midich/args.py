import argparse
from .config import DEFAULT_SLEEP_TIME, DEFAULT_GUIDED, DEFAULT_ADD_COLOR, DEFAULT_SUITS_COLOR

def parse_args():
    parser = argparse.ArgumentParser(description="Midich расклад")
    parser.add_argument('--guided', action='store_true', default=DEFAULT_GUIDED)
    parser.add_argument('--sleep', type=float, default=DEFAULT_SLEEP_TIME)

    parser.add_argument('--add-color', type=str, default=DEFAULT_ADD_COLOR)
    for suit in DEFAULT_SUITS_COLOR:
        parser.add_argument(
            f'--{suit.lower()}-color',
            type=str,
            default=DEFAULT_SUITS_COLOR[suit],
            help=f'Цвет для масти {suit}'
        )

    return parser.parse_args()