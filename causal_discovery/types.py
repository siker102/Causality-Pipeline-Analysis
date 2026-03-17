from typing import Protocol


class ProgressCallback(Protocol):
    def __call__(self, message: str, progress: float = 0.0) -> None:
        ...
