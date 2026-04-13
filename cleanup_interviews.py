"""
Database Cleanup Tool
=====================
Remove interviews by student name or clear anonymous records.

Usage:
    python cleanup_interviews.py "Student Name"      # Remove by name
    python cleanup_interviews.py --anonymous          # Remove anonymous/empty
    python cleanup_interviews.py --list               # List all interviews
"""

import sqlite3
import sys
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interviews.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def list_interviews():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT i.id, i.student_name, i.interview_type, i.created_at, 
               s.dominant_sentiment
        FROM interviews i
        LEFT JOIN interview_summary s ON i.id = s.interview_id
        ORDER BY i.created_at DESC
    """)
    rows = c.fetchall()
    print(f"\n{'ID':<6} {'Student':<30} {'Type':<12} {'Sentiment':<12} {'Date'}")
    print("-" * 85)
    for r in rows:
        name = r[1] or "(Anonymous)"
        print(f"{r[0]:<6} {name:<30} {r[2] or '--':<12} {r[4] or '--':<12} {r[3] or '--'}")
    print(f"\nTotal: {len(rows)} interviews")
    conn.close()


def delete_interviews(ids, conn):
    if not ids:
        return
    ph = ','.join('?' * len(ids))
    c = conn.cursor()
    c.execute(f'DELETE FROM topic_classifications WHERE interview_id IN ({ph})', ids)
    c.execute(f'DELETE FROM interview_summary WHERE interview_id IN ({ph})', ids)
    c.execute(f'DELETE FROM analysis_results WHERE interview_id IN ({ph})', ids)
    c.execute(f'DELETE FROM transcripts WHERE interview_id IN ({ph})', ids)
    c.execute(f'DELETE FROM interviews WHERE id IN ({ph})', ids)
    conn.commit()


def remove_by_name(name):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, student_name FROM interviews WHERE student_name LIKE ?", (f'%{name}%',))
    rows = c.fetchall()

    if not rows:
        print(f'No interviews found for "{name}"')
        conn.close()
        return

    print(f'\nFound {len(rows)} interview(s) matching "{name}":')
    for r in rows:
        print(f'  ID {r[0]}: {r[1]}')

    confirm = input(f'\nDelete all {len(rows)}? (y/n): ').strip().lower()
    if confirm == 'y':
        delete_interviews([r[0] for r in rows], conn)
        print(f'Deleted {len(rows)} interview(s) and all related data.')
    else:
        print('Cancelled.')

    c.execute('SELECT COUNT(*) FROM interviews')
    print(f'Remaining interviews: {c.fetchone()[0]}')
    conn.close()


def remove_anonymous():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM interviews WHERE student_name IS NULL OR student_name = ''")
    ids = [r[0] for r in c.fetchall()]

    if not ids:
        print('No anonymous interviews found.')
        conn.close()
        return

    print(f'Found {len(ids)} anonymous interview(s).')
    confirm = input(f'Delete all {len(ids)}? (y/n): ').strip().lower()
    if confirm == 'y':
        delete_interviews(ids, conn)
        print(f'Deleted {len(ids)} anonymous interview(s).')
    else:
        print('Cancelled.')

    c.execute('SELECT COUNT(*) FROM interviews')
    print(f'Remaining interviews: {c.fetchone()[0]}')
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    arg = sys.argv[1]

    if arg == '--list':
        list_interviews()
    elif arg == '--anonymous':
        remove_anonymous()
    else:
        remove_by_name(arg)
