"""Game-specific exceptions."""


class GameException(Exception):
    """Base exception for game errors."""
    pass


class InvalidActionError(GameException):
    """Raised when action is not legal in current state."""
    pass


class InsufficientResourcesError(GameException):
    """Raised when player lacks required resources."""
    pass


class InvalidPlacementError(GameException):
    """Raised when building placement violates rules."""
    pass


class GameOverError(GameException):
    """Raised when attempting action in finished game."""
    pass
