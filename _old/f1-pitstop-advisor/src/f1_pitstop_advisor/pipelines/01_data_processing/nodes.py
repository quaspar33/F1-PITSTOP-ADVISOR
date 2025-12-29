import datetime
import pandas as pd
import pit_advisor.gather_data as gd
import pit_advisor.data_creation as dc


from typing import List
from fastf1.core import Session

def load_sessions() -> List[Session]:
    sessions = gd.get_sessions(datetime.datetime(2025, 10, 20))
    sessions = [sessions[0]]
    return gd.load_sessions(sessions)


def prepare_data(sessions: List[Session]) -> pd.DataFrame:
    return dc._get_refined_lap_data_with_z_score(sessions)
