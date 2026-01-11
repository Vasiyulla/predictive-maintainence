import sqlite3, pandas as pd
conn = sqlite3.connect('database/predictxai.db')
pd.read_sql_query("SELECT * FROM users", conn)