import pandas as pd

# Try to read the file
try:
    # We use encoding='latin-1' because this specific dataset has some special characters
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("✅ Success! Dataset loaded.")
    print(df.head()) # Shows the first 5 rows
except FileNotFoundError:
    print("❌ Error: The file 'spam.csv' was not found.")
    print("Make sure the file is in the exact same folder as your python script.")