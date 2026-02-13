"""Action-specific exceptions."""


class ActionException(Exception):
    """Base exception for action errors."""
    pass


class ActionNotAllowedError(ActionException):
    """Action not allowed in current game phase."""
    pass


class InvalidParametersError(ActionException):
    """Action parameters are invalid."""
    pass
