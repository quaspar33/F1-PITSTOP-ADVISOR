import pandas as pd


import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

from fastf1.core import Session
from typing import List


def _get_sessions(cutoff_year: int) -> List[Session]:
    warnings.filterwarnings('ignore')

    sessions = []

    current_year = datetime.now().year
    years = range(2022, current_year + 1)

    for year in years:
        print(f"\nAnalizuję rok {year}...")

        try:
            race_calendar = f1.get_event_schedule(year)
            races = race_calendar[race_calendar['EventFormat'] == 'conventional']

            if year == current_year:
                today = pd.Timestamp.now()
                races = races[races['EventDate'] < today]

            if races.empty:
                print(f"Brak wyścigów dla roku {year}")
                continue

            print(f"Znaleziono {len(races)} wyścigów dla roku {year}")

            for idx, race in races.iterrows():
                race_name = race['EventName']
                race_round = race['RoundNumber']

                try:
                    print(f"  Pobieram dane dla wyścigu: {race_name} (Runda {race_round})")
                    session = f1.get_session(year, race_round, 'R')
                    sessions.append(session)
                except Exception as e:
                    print(f"    Błąd podczas pobierania danych dla {race_name}: {e}")
        except Exception as e:
            print(f"Błąd podczas pobierania kalendarza dla roku {year}: {e}")
    return sessions

def extract_historical_data(cutoff_year: int) -> pd.DataFrame:
    sessions = _get_sessions(cutoff_year)
    pass

def extract_race_data(cutoff_year: int) -> pd.DataFrame:
    sessions = _get_sessions(cutoff_year)
    pass