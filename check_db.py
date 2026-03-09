import database as db

print('Checking database...')
conn = db.get_connection()
cursor = conn.cursor()

# Check tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = [r[0] for r in cursor.fetchall()]
print('Tables:', tables)

# Check if topic_classifications exists
if 'topic_classifications' in tables:
    cursor.execute('SELECT COUNT(*) FROM topic_classifications')
    count = cursor.fetchone()[0]
    print(f'Topic classifications count: {count}')
    
    if count > 0:
        cursor.execute('SELECT * FROM topic_classifications LIMIT 1')
        row = cursor.fetchone()
        print('Sample row:', dict(row))
else:
    print('topic_classifications table does NOT exist!')

conn.close()
