from .args import parse_args
from .core.rules import midich
from .utils import get_color
from termcolor import colored

def main():
    args = parse_args()

    suit_colors = {
        "DIAMOND": args.diamond_color,
        "HEARTS": args.hearts_color,
        "CLUBS": args.clubs_color,
        "SPADES": args.spades_color,
    }

    result = midich(guided=args.guided, sleep_time=args.sleep, suit_colors=suit_colors, add_color=args.add_color)
    if not args.guided:
        print(" ".join(map(lambda card : colored(str(card), get_color(card, suit_colors)), result)))
    else:
        print()

if __name__ == "__main__":
    main()
