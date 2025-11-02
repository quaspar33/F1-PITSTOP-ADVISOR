from pyexpat.errors import messages

import pandas as pd
import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
from fastf1.core import Session
from typing import List



def get_sessions(cutoff_date: datetime) -> List[Session]:
    warnings.filterwarnings('ignore')

    sessions = []

    start_year = 2022
    end_year = cutoff_date.year
    years = range(start_year, end_year + 1)

    for year in years:
        print(f"\nAnalizuję rok {year}...")

        try:
            race_calendar = f1.get_event_schedule(year)
            races = race_calendar[race_calendar['EventFormat'] == 'conventional']

            cutoff_timestamp = pd.Timestamp(cutoff_date)
            races = races[races['EventDate'] < cutoff_timestamp]

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
    sessions = get_sessions(cutoff_year)
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

def extract_pitstop_data(cutoff_year: int) -> pd.DataFrame:
    sessions = get_sessions(cutoff_year)
    all_pitstops_data = []

    for session in sessions:
        try:
            session.load(laps=True)

            year = session.event['EventDate'].year
            race_name = session.event['EventName']
            race_round = session.event['RoundNumber']

            print(f"  Analizuję pit stopy dla wyścigu: {race_name} {year}")

            drivers = session.drivers

            for driver in drivers:
                try:
                    driver_info = session.get_driver(driver)
                    driver_name = driver_info['Abbreviation']
                    team = driver_info['TeamName']

                    driver_laps = session.laps.pick_driver(driver)

                    for _, lap in driver_laps[driver_laps['PitInTime'].notna()].iterrows():
                        lap_number = lap['LapNumber']

                        previous_compound = None
                        previous_laps = driver_laps[driver_laps['LapNumber'] < lap_number]
                        if not previous_laps.empty:
                            previous_compound = previous_laps.iloc[-1]['Compound']

                        next_compound = None
                        next_laps = driver_laps[driver_laps['LapNumber'] > lap_number]
                        if not next_laps.empty:
                            next_compound = next_laps.iloc[0]['Compound']

                        pitstop_data = {
                            'Year': year,
                            'Race': race_name,
                            'Round': race_round,
                            'Driver': driver_name,
                            'Team': team,
                            'LapNumber': lap_number,
                            'CompoundChangedFrom': previous_compound,
                            'CompoundChangedTo': next_compound
                        }

                        all_pitstops_data.append(pitstop_data)

                except Exception as e:
                    print(f"    Błąd podczas analizy danych kierowcy {driver} w {race_name}: {e}")

        except Exception as e:
            race_info = f"{session.event.year} {session.event['EventName']}"
            print(f"    Błąd podczas analizy pit stopów dla {race_info}: {e}")

    if all_pitstops_data:
        return pd.DataFrame(all_pitstops_data)
    else:
        raise ValueError("Nie znaleziono danych o pit stopach dla podanego okresu.")
