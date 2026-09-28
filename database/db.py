import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dineiq.db')

def init_db():
    """Database aur tables banayein agar nahi hain."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Password ko secure hash mein convert karein."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, role='analyst'):
    """Naya user register karein."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists!"

def authenticate_user(username, password):
    """User ko authenticate karein."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, role FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    if user:
        return True, {'username': user[0], 'role': user[1]}
    return False, None

def get_all_users():
    """Saare users ki list (admin ke liye)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, role, created_at FROM users")
    users = c.fetchall()
    conn.close()
    return users

# Default admin user create karein (pehli baar)
def create_default_admin():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE username='admin'")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  ('admin', hash_password('dineiq123'), 'admin'))
        conn.commit()
    conn.close()

# Initialize
init_db()
create_default_admin()