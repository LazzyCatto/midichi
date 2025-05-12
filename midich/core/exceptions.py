class InvalidCardValue(Exception):
    def __init__(self, value, *args):
        super().__init__(f"Invalid card value: {value}", *args)
