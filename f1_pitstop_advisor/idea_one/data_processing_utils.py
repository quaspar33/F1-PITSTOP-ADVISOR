import pandas as pd
import numpy as np

from fastf1.core import Session


def get_lap_data_with_weather(session: Session) -> pd.DataFrame:
    # Prepare raw data
    weather_data: pd.DataFrame = session.weather_data.copy() # type: ignore
    laps: pd.DataFrame = session.laps.copy()

    # Drop laps with missing lap time
    laps.dropna(subset=["LapTime"], ignore_index=True, inplace=True)

    # Prepare an indexer that indexes weather data for every lap
    weather_for_intervals = weather_data.loc[:, ["Time"]].copy()
    weather_for_intervals["EndTime"] = weather_for_intervals["Time"].shift(-1)
    weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "EndTime"] = (
        weather_for_intervals.loc[weather_for_intervals.last_valid_index(), "Time"] + np.timedelta64(1, "m") # type: ignore
    )
    weather_interval_index = pd.IntervalIndex.from_arrays(weather_for_intervals["Time"], weather_for_intervals["EndTime"], closed="both")
    weather_indexer, _ = weather_interval_index.get_indexer_non_unique(laps["Time"]) # type: ignore
    weather_data["TmpJoinIndex"] = weather_data.index
    laps["TmpJoinIndex"] = pd.Series(weather_indexer)

    # Merge laps with weather data
    data = laps.merge(weather_data, on="TmpJoinIndex", suffixes=("", "_y"))

    data.drop(["TmpJoinIndex", "Time_y"], axis="columns", inplace=True)
    return data