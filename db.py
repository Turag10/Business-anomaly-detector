import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return hashed, salt

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Cases investigation table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS investigations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        account_id TEXT,
        created_at TEXT NOT NULL,
        analyst TEXT NOT NULL,
        status TEXT NOT NULL,
        risk_score REAL NOT NULL,
        verdict TEXT NOT NULL,
        notes TEXT
    )
    """)
    
    # User authentication table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'Fraud Operations Analyst',
        created_at TEXT NOT NULL
    )
    """)
    
    conn.commit()
    conn.close()
    seed_default_users()

def seed_default_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM users")
    row = cursor.fetchone()
    if row and row['count'] == 0:
        # Seed default admin / demo investigator
        h_admin, s_admin = hash_password("admin123")
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("admin", "admin@bankops.internal", h_admin, s_admin, "Alex Vance", "Lead Risk Architect", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        h_analyst, s_analyst = hash_password("investigator123")
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("analyst", "analyst@bankops.internal", h_analyst, s_analyst, "Sarah Jenkins", "Fraud Operations Analyst", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
    conn.close()

def create_user(username, email, password, full_name, role="Fraud Operations Analyst"):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if username or email already exists
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username.strip().lower(), email.strip().lower()))
    if cursor.fetchone():
        conn.close()
        return False, "Username or email is already registered."
        
    pwd_hash, salt = hash_password(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.strip().lower(), email.strip().lower(), pwd_hash, salt, full_name.strip(), role, created_at))
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except Exception as e:
        conn.close()
        return False, f"Registration error: {str(e)}"

def verify_user(username_or_email, password):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    target = username_or_email.strip().lower()
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (target, target))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None, "Invalid username or password."
        
    pwd_hash, _ = hash_password(password, user['salt'])
    if pwd_hash == user['password_hash']:
        return {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'full_name': user['full_name'],
            'role': user['role']
        }, "Authenticated successfully."
    else:
        return None, "Invalid username or password."

def add_case(transaction_id, account_id, analyst, status, risk_score, verdict, notes):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO investigations (transaction_id, account_id, created_at, analyst, status, risk_score, verdict, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (transaction_id, account_id, created_at, analyst, status, float(risk_score), verdict, notes))
    conn.commit()
    conn.close()

def get_all_cases():
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM investigations ORDER BY id DESC", conn)
    conn.close()
    return df

def update_case(case_id, status, notes):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE investigations
    SET status = ?, notes = ?
    WHERE id = ?
    """, (status, notes, case_id))
    conn.commit()
    conn.close()

def delete_case(case_id):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM investigations WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized with user authentication at", DB_PATH)
