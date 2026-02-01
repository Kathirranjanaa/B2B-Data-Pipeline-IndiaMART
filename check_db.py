import sqlite3
import pandas as pd

conn = sqlite3.connect("products.db")

# Show tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables:\n", tables)

# Count rows
count = pd.read_sql("SELECT COUNT(*) as total_rows FROM products;", conn)
print("\nRow count:\n", count)

# Preview
sample = pd.read_sql("SELECT * FROM products LIMIT 5;", conn)
print("\nSample:\n", sample)

conn.close()
