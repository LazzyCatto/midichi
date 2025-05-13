class InvalidCardValue(Exception):
    def __init__(self, value, *args):
        super().__init__(f"Invalid card value: {value}", *args)

class EmptyDeckAccess(Exception):
    def __init__(self, *args):
        super().__init__(f"Empty deck access", *args)

class GuidedMissingParameters(Exception):
    def __init__(self, missing_params, *args):

        super().__init__("guided=True, missing parameters.\n" + " ".join(missing_params) + " are not set", *args)