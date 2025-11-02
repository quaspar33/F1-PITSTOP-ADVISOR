import pandas as pd


from typing import List
from fastf1.core import Session

def load_sessions() -> List[Session]:
    raise NotImplementedError()


def prepare_data(sessions: List[Session]) -> pd.DataFrame:
    raise NotImplementedError()
