import sys
from f1_pitstop_advisor import gather_data

if len(sys.argv) > 1:
    path = sys.argv[0]
else:
    path = "ig_sessions.pickle"

sessions = gather_data._get_sessions(2022)

# Load data to cache
for i, session in zip(range(len(sessions)), sessions):
    try:
        session.load()
        print(f"Loaded session {i + 1} of {len(sessions)}")
    except RuntimeError as e:
        print(e)
        print(f"Failed to load session {i + 1} of {len(sessions)}")

# Save sessions to pickle for quick reuse
import pickle

with open(path, "wb") as file:
    pickle.dump(sessions, file)


