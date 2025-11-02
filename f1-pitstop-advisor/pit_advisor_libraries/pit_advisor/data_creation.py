import pit_advisor.data_processing_utils as utils

import pandas as pd
import numpy as np
import pickle

from fastf1.core import Session
from typing import Dict, Iterable, List

def _get_refined_lap_data_with_z_score(sessions: List[Session]) -> pd.DataFrame:
    if not sessions:
        raise ValueError(f"Parameter \"sessions\" may not be an empty list.")
    data_list = []
    for session in sessions:
        session_data = utils.get_lap_data_with_weather(session)
        utils.add_z_score_for_laps(session_data, inplace=True)
        session_data = session_data.convert_dtypes()
        data_list.append(session_data)

    data = pd.concat(data_list, ignore_index=True)

    # Add a feature determining whether there was a pit stop performed during each lap
    utils.add_is_pit_lap(data, inplace=True)

    # Select only relevant columns for further processing
    selected_columns = [
        "LapTimeZScore",
        "IsPitLap",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "LapNumber",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed"
    ]
    filtered_data = data.loc[:, selected_columns]

    # Convert categorical data to boolean values
    final_data = pd.get_dummies(filtered_data)
    return final_data
    

def get_refined_lap_data_with_z_score_by_circuit(sessions: List[Session]) -> Dict[str, pd.DataFrame]:
    circuits_and_sessions = {}
    for session in sessions:
        circuit = session.session_info["Meeting"]["Circuit"]["ShortName"]
        if circuit not in circuits_and_sessions.keys():
            circuits_and_sessions[circuit] = []
        circuits_and_sessions[circuit].append(session)

    dfs = {}
    for circuit, sessions in circuits_and_sessions.items():
        dfs[circuit] = _get_refined_lap_data_with_z_score(sessions)

    return dfs

