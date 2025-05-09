import pandas as pd
import numpy as np
import urllib3

import fastf1
import fastf1.core

from typing import Dict


def _get_session_data(cutoff_year: int, circuit_key: int | None = None) -> pd.DataFrame:
    http = urllib3.PoolManager()
    request = f"https://api.openf1.org/v1/sessions?session_type=Race&year>={cutoff_year}"
    if circuit_key is not None:
        request += f"&circuit_key={circuit_key}"
    response = http.request("GET", request)
    return pd.DataFrame(response.json())


def get_sessions(cutoff_year: int, circuit_key: int | None = None) -> Dict[int, fastf1.core.Session]:
    session_data = _get_session_data(cutoff_year, circuit_key)
    sessions = dict[int, fastf1.core.Session]()

    def load_telemetry_data(session_row: pd.Series) -> None:
        session_key = session_row["session_key"]
        year = session_row["year"]
        gp = session_row["country_name"]
        identifier = session_row["session_name"]
        sessions[session_key] = fastf1.get_session(year, gp, identifier)

    session_data.apply(load_telemetry_data, axis=1) # type: ignore

    for session in sessions.values():
        session.load()
    return sessions



