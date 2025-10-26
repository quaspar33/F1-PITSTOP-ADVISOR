import sys
import os
import pickle
import gather_data
from datetime import datetime

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "f1_pitstop_advisor"

if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} not found.")
else:
    print(f"File {path} found.")

file_name = "ig_sessions.pickle"
final_path = os.path.join(path, file_name)

sessions = gather_data._get_sessions(datetime(2025, 10, 20))

# Load data to cache
for i, session in zip(range(len(sessions)), sessions):
    try:
        session.load()
        print(f"Loaded session {i + 1} of {len(sessions)}")
    except RuntimeError as e:
        print(e)
        print(f"Failed to load session {i + 1} of {len(sessions)}")

# Save sessions to pickle for quick reuse
with open(final_path, "wb") as file:
    pickle.dump(sessions, file)


