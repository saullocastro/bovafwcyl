import pickle

# Load the pickle file
with open("GA_0500_kN_individuals.pickle", "rb") as f:  # Change filename as needed
    data = pickle.load(f)

import pandas as pd

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("output_500.csv", index=False)

print("Pickle file converted to CSV successfully!")
