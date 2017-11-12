import sqlite3

with sqlite3.connect('reddit.db') as conn:
    conn.execute('drop table sentiment_table_values')
    conn.commit()