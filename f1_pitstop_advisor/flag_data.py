import pandas as pd
import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

target_flags = ['YELLOW', 'DOUBLE YELLOW', 'RED']

all_races_data = []
races_with_flags = {flag: [] for flag in target_flags}
total_races = 0

current_year = datetime.now().year
current_month = datetime.now().month

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
                session.load(messages=True)

                messages_df = session.race_control_messages

                flag_counts = {}
                for flag in target_flags:
                    count = sum(messages_df['Flag'] == flag)
                    flag_counts[flag] = count

                    if count > 0:
                        races_with_flags[flag].append(f"{year} {race_name}")

                race_data = {
                    'Year': year,
                    'Race': race_name,
                    'Round': race_round
                }
                race_data.update(flag_counts)
                all_races_data.append(race_data)
                total_races += 1

            except Exception as e:
                print(f"    Błąd podczas pobierania danych dla {race_name}: {e}")

    except Exception as e:
        print(f"Błąd podczas pobierania kalendarza dla roku {year}: {e}")

if all_races_data:
    races_df = pd.DataFrame(all_races_data)

    print("\n--- STATYSTYKI FLAG ---")
    for flag in target_flags:
        total_occurrences = races_df[flag].sum()
        races_with_flag = sum(races_df[flag] > 0)
        probability = races_with_flag / total_races if total_races > 0 else 0

        print(f"\n{flag} FLAGA:")
        print(f"  Łączna liczba wystąpień: {total_occurrences}")
        print(f"  Wyścigi z tą flagą: {races_with_flag} z {total_races} ({probability:.2%})")
        print(f"  Średnia wystąpień na wyścig: {total_occurrences / total_races:.2f}")

    probabilities = [sum(races_df[flag] > 0) / total_races for flag in target_flags]

    yearly_probabilities = {}
    for year in years:
        year_races = races_df[races_df['Year'] == year]
        year_total = len(year_races)
        if year_total > 0:
            yearly_probabilities[year] = {
                flag: sum(year_races[flag] > 0) / year_total for flag in target_flags
            }

    yearly_prob_data = []
    for year, probs in yearly_probabilities.items():
        year_data = {'Year': year}
        year_data.update(probs)
        yearly_prob_data.append(year_data)

    yearly_prob_df = pd.DataFrame(yearly_prob_data)

    print("\n--- SZCZEGÓŁOWE DANE KAŻDEGO WYŚCIGU ---")
    print(races_df)

    for flag in target_flags:
        races_with_flags[flag].sort()

    races_df.to_csv('f1_flags_analysis.csv', index=False)
    print("\nZapisano szczegółowe dane do pliku 'f1_flags_analysis.csv'")

else:
    print("Nie znaleziono żadnych danych.")