import os


def format_secs(secs: float) -> str:
    return "{0:.3f} secs".format(secs)


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
