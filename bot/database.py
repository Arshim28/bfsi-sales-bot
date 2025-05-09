import os
import uuid
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Database path
DATABASE_PATH = Path(__file__).parent.parent / "data" / "bfsi_bot.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        # Ensure data directory exists
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to SQLite database
        conn = sqlite3.connect(str(DATABASE_PATH))
        
        # Enable dictionary cursor
        conn.row_factory = sqlite3.Row
        
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor():
    """Context manager for database cursors with automatic commit/rollback."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

# FastAPI dependency
def get_db():
    """Dependency for FastAPI to get a database connection."""
    with get_db_connection() as conn:
        yield conn

# Database initialization with table creation functions
def create_users_table():
    """Create users table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            api_key TEXT UNIQUE NOT NULL
        )
        """)

def create_documents_table():
    """Create documents table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            document_type TEXT NOT NULL,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            processed INTEGER DEFAULT 0,
            processed_at TEXT,
            content_preview TEXT,
            output_path TEXT,
            processing_error TEXT
        )
        """)

def create_generations_table():
    """Create generations table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            knowledge_base_id INTEGER REFERENCES documents(id),
            agent_persona_id INTEGER REFERENCES documents(id),
            status TEXT DEFAULT 'pending',
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            client_types_count INTEGER DEFAULT 0,
            questions_count INTEGER DEFAULT 0,
            questions_per_client INTEGER DEFAULT 50,
            output_directory TEXT,
            error_message TEXT,
            analysis_path TEXT,
            analysis_completed INTEGER DEFAULT 0,
            analysis_completed_at TEXT,
            analysis_error TEXT
        )
        """)

def create_client_types_table():
    """Create client_types table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id INTEGER REFERENCES generations(id),
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            question_count INTEGER DEFAULT 0,
            output_file TEXT
        )
        """)

def init_db():
    """Initialize the database by creating all tables."""
    create_users_table()
    create_documents_table()
    create_generations_table()
    create_client_types_table()
    print("Database tables created.")

# User operations
def create_user(username, email, password_hash):
    """Create a new user."""
    api_key = str(uuid.uuid4())
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, api_key)
        VALUES (?, ?, ?, ?)
        """, (username, email, password_hash, api_key))
        
        # Get the last inserted id
        cursor.execute("SELECT last_insert_rowid()")
        user_id = cursor.fetchone()[0]
        
        # Get the new user data
        cursor.execute("""
        SELECT id, username, email, created_at, is_active, api_key
        FROM users WHERE id = ?
        """, (user_id,))
        return dict(cursor.fetchone())

def get_user_by_username(username):
    """Get user by username."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE username = ?
        """, (username,))
        result = cursor.fetchone()
        return dict(result) if result else None

def get_user_by_email(email):
    """Get user by email."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE email = ?
        """, (email,))
        result = cursor.fetchone()
        return dict(result) if result else None

def get_user_by_api_key(api_key):
    """Get user by API key."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE api_key = ?
        """, (api_key,))
        result = cursor.fetchone()
        return dict(result) if result else None

# Document operations
def create_document(user_id, filename, file_path, document_type):
    """Create a new document record."""
    now = datetime.utcnow().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO documents (user_id, filename, file_path, document_type, uploaded_at)
        VALUES (?, ?, ?, ?, ?)
        """, (user_id, filename, file_path, document_type, now))
        
        # Get the last inserted id
        cursor.execute("SELECT last_insert_rowid()")
        doc_id = cursor.fetchone()[0]
        
        # Get the new document data
        cursor.execute("""
        SELECT id, user_id, filename, file_path, document_type, uploaded_at, processed, processed_at, content_preview
        FROM documents WHERE id = ?
        """, (doc_id,))
        return dict(cursor.fetchone())

def get_document(document_id, user_id=None):
    """Get document by ID, optionally filtering by user_id."""
    with get_db_cursor() as cursor:
        if user_id:
            cursor.execute("""
            SELECT * FROM documents WHERE id = ? AND user_id = ?
            """, (document_id, user_id))
        else:
            cursor.execute("""
            SELECT * FROM documents WHERE id = ?
            """, (document_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def get_documents_by_user(user_id, document_type=None):
    """Get all documents for a user, optionally filtering by type."""
    with get_db_cursor() as cursor:
        if document_type:
            cursor.execute("""
            SELECT * FROM documents WHERE user_id = ? AND document_type = ?
            """, (user_id, document_type))
        else:
            cursor.execute("""
            SELECT * FROM documents WHERE user_id = ?
            """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def update_document_processed(document_id, output_path, content_preview=None):
    """Update a document's processed status."""
    now = datetime.utcnow().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE documents 
        SET processed = 1, processed_at = ?, output_path = ?, content_preview = ?
        WHERE id = ?
        """, (now, output_path, content_preview, document_id))
        
        # Get the updated document
        cursor.execute("""
        SELECT * FROM documents WHERE id = ?
        """, (document_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def update_document_error(document_id, error_message):
    """Update a document with processing error."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE documents 
        SET processing_error = ?
        WHERE id = ?
        """, (error_message, document_id))
        
        # Get the updated document
        cursor.execute("""
        SELECT * FROM documents WHERE id = ?
        """, (document_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def delete_document(document_id, user_id):
    """Delete a document by ID and user_id."""
    with get_db_cursor() as cursor:
        # Get the document first
        cursor.execute("""
        SELECT * FROM documents
        WHERE id = ? AND user_id = ?
        """, (document_id, user_id))
        result = cursor.fetchone()
        
        if not result:
            return None
            
        # Then delete it
        cursor.execute("""
        DELETE FROM documents
        WHERE id = ? AND user_id = ?
        """, (document_id, user_id))
        
        return dict(result)

# Generation operations
def create_generation(user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory):
    """Create a new generation record."""
    now = datetime.utcnow().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO generations 
        (user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory, 'pending', now))
        
        # Get the last inserted id
        cursor.execute("SELECT last_insert_rowid()")
        gen_id = cursor.fetchone()[0]
        
        # Get the new generation data
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (gen_id,))
        return dict(cursor.fetchone())

def update_generation_status(generation_id, status, error_message=None):
    """Update a generation's status."""
    with get_db_cursor() as cursor:
        if status == 'completed':
            now = datetime.utcnow().isoformat()
            cursor.execute("""
            UPDATE generations 
            SET status = ?, completed_at = ?
            WHERE id = ?
            """, (status, now, generation_id))
        elif status == 'failed':
            cursor.execute("""
            UPDATE generations 
            SET status = ?, error_message = ?
            WHERE id = ?
            """, (status, error_message, generation_id))
        else:
            cursor.execute("""
            UPDATE generations 
            SET status = ?
            WHERE id = ?
            """, (status, generation_id))
        
        # Get the updated generation
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def update_generation_counts(generation_id, client_types_count, questions_count):
    """Update a generation's counts."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET client_types_count = ?, questions_count = ?
        WHERE id = ?
        """, (client_types_count, questions_count, generation_id))
        
        # Get the updated generation
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def get_generation(generation_id, user_id=None):
    """Get generation by ID, optionally filtering by user_id."""
    with get_db_cursor() as cursor:
        if user_id:
            cursor.execute("""
            SELECT g.*, u.username FROM generations g
            JOIN users u ON g.user_id = u.id
            WHERE g.id = ? AND g.user_id = ?
            """, (generation_id, user_id))
        else:
            cursor.execute("""
            SELECT g.*, u.username FROM generations g
            JOIN users u ON g.user_id = u.id
            WHERE g.id = ?
            """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def get_generations_by_user(user_id):
    """Get all generations for a user."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM generations WHERE user_id = ?
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def delete_generation(generation_id, user_id):
    """Delete a generation by ID and user_id."""
    with get_db_cursor() as cursor:
        # First delete related client types
        cursor.execute("""
        DELETE FROM client_types
        WHERE generation_id = ?
        """, (generation_id,))
        
        # Get the generation first
        cursor.execute("""
        SELECT * FROM generations
        WHERE id = ? AND user_id = ?
        """, (generation_id, user_id))
        result = cursor.fetchone()
        
        if not result:
            return None
            
        # Then delete it
        cursor.execute("""
        DELETE FROM generations
        WHERE id = ? AND user_id = ?
        """, (generation_id, user_id))
        
        return dict(result)

# Client type operations
def create_client_type(generation_id, name, description, question_count, output_file):
    """Create a new client type record."""
    now = datetime.utcnow().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO client_types 
        (generation_id, name, description, question_count, output_file, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (generation_id, name, description, question_count, output_file, now))
        
        # Get the last inserted id
        cursor.execute("SELECT last_insert_rowid()")
        client_type_id = cursor.fetchone()[0]
        
        # Get the new client type data
        cursor.execute("""
        SELECT * FROM client_types WHERE id = ?
        """, (client_type_id,))
        return dict(cursor.fetchone())

def get_client_types_by_generation(generation_id):
    """Get all client types for a generation."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM client_types WHERE generation_id = ?
        """, (generation_id,))
        return [dict(row) for row in cursor.fetchall()]

# Analysis operations
def update_generation_analysis(generation_id, analysis_path):
    """Update a generation with analysis information."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_path = ?
        WHERE id = ?
        """, (analysis_path, generation_id))
        
        # Get the updated generation
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def complete_generation_analysis(generation_id):
    """Mark a generation's analysis as completed."""
    now = datetime.utcnow().isoformat()
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_completed = 1, analysis_completed_at = ?
        WHERE id = ?
        """, (now, generation_id))
        
        # Get the updated generation
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

def update_generation_analysis_error(generation_id, error_message):
    """Update a generation with analysis error."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_error = ?
        WHERE id = ?
        """, (error_message, generation_id))
        
        # Get the updated generation
        cursor.execute("""
        SELECT * FROM generations WHERE id = ?
        """, (generation_id,))
        result = cursor.fetchone()
        return dict(result) if result else None

if __name__ == "__main__":
    # When run directly, initialize the database
    init_db()
    print("Database tables created.")