from pyexpat.errors import messages

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
    years = range(cutoff_year, current_year + 1)

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

def extract_flag_data(cutoff_year: int) -> pd.DataFrame:
    target_flags = ['YELLOW', 'DOUBLE YELLOW', 'RED']
    sessions = _get_sessions(cutoff_year)
    all_races_data = []

    for session in sessions:
        try:
            session.load(messages=True)

            race_data = {
                'Year': session.event['EventDate'].year,
                'Race': session.event['EventName'],
                'Round': session.event['RoundNumber']
            }

            messages_df = session.race_control_messages
            for flag in target_flags:
                count = sum(messages_df['Flag'] == flag)
                race_data[flag] = count

            all_races_data.append(race_data)

        except Exception as e:
            race_info = f"{session.event.year} {session.event['EventName']}"
            print(f"    Błąd podczas analizy danych dla {race_info}: {e}")

    if all_races_data:
        return pd.DataFrame(all_races_data)
    else:
        raise ValueError("Nie znaleziono danych do utworzenia DataFramu.")
