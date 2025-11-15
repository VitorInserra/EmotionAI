from pydantic import BaseModel
from typing import List


class VRData(BaseModel):
    start_stamp: str
    end_stamp: str
    eye_id: str
    text_version: int
    seen_words: dict[str, float]

