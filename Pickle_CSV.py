import pickle
import pandas as pd

# Load the pickle file (the optimum one)
with open("GA_0050_kN_best_individual.pickle", "rb") as f: #change as needed
    data = pickle.load(f)

# Convert the dict to a DataFrame
df = pd.DataFrame([data])  # wrap in list of one dict

# Save to CSV
df.to_csv("50_best.csv", index=False)

print("Converted successfully!")

# Load the pickle file (complete results)
with open("GA_0050_kN_individuals.pickle", "rb") as f:  # Change filename as needed
    data = pickle.load(f)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("output_50.csv", index=False)

print("Pickle file converted to CSV successfully!")
