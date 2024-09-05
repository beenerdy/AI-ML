import sqlite3

import pandas as pd

# Connect to the SQLite database
db_path = "/Users/beenerdy/Music/Engine Library/Database2/hm.db"  # Replace with your .db file path
conn = sqlite3.connect(db_path)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# SQL query to get the first row from the Track table
query = "SELECT * FROM Track LIMIT 1;"

# Execute the query and fetch the first row
cursor.execute(query)
row = cursor.fetchone()

# Get column names from the Track table
column_names = [description[0] for description in cursor.description]

# Convert the row to a DataFrame
df = pd.DataFrame([row], columns=column_names)

# Save the DataFrame to a CSV file
csv_path = "first_track_item.csv"  # Output CSV file path
df.to_csv(csv_path, index=False)

# Close the database connection
conn.close()

print(f"The first row from the Track table has been saved to {csv_path}.")
