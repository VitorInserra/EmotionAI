from pydantic import BaseModel
from typing import List


class VRData(BaseModel):
    batch_id: int
    start_stamp: str
    end_stamp: str
    eye_id: str
    seen_words: str
    text_version: int

