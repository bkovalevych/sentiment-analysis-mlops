from typing import NamedTuple


class ReviewItem(NamedTuple):
    id: int
    overall: float
    reviewText: str
    verified: bool
    summary: str
