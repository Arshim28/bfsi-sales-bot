import os
import uuid
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from psycopg.conninfo import make_conninfo

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Get database URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/bfsi_bot")

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        # Connect with connection parameters
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
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
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            api_key VARCHAR(64) UNIQUE NOT NULL
        )
        """)

def create_documents_table():
    """Create documents table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            filename VARCHAR(255) NOT NULL,
            file_path VARCHAR(512) NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT FALSE,
            processed_at TIMESTAMP,
            content_preview TEXT,
            output_path VARCHAR(512),
            processing_error TEXT
        )
        """)

def create_generations_table():
    """Create generations table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            knowledge_base_id INTEGER REFERENCES documents(id),
            agent_persona_id INTEGER REFERENCES documents(id),
            status VARCHAR(20) DEFAULT 'pending',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            client_types_count INTEGER DEFAULT 0,
            questions_count INTEGER DEFAULT 0,
            questions_per_client INTEGER DEFAULT 50,
            output_directory VARCHAR(512),
            error_message TEXT,
            analysis_path VARCHAR(512),
            analysis_completed BOOLEAN DEFAULT FALSE,
            analysis_completed_at TIMESTAMP,
            analysis_error TEXT
        )
        """)

def create_client_types_table():
    """Create client_types table if it doesn't exist."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS client_types (
            id SERIAL PRIMARY KEY,
            generation_id INTEGER REFERENCES generations(id),
            name VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            question_count INTEGER DEFAULT 0,
            output_file VARCHAR(512)
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
        VALUES (%s, %s, %s, %s)
        RETURNING id, username, email, created_at, is_active, api_key
        """, (username, email, password_hash, api_key))
        return cursor.fetchone()

def get_user_by_username(username):
    """Get user by username."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE username = %s
        """, (username,))
        return cursor.fetchone()

def get_user_by_email(email):
    """Get user by email."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE email = %s
        """, (email,))
        return cursor.fetchone()

def get_user_by_api_key(api_key):
    """Get user by API key."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM users WHERE api_key = %s
        """, (api_key,))
        return cursor.fetchone()

# Document operations
def create_document(user_id, filename, file_path, document_type):
    """Create a new document record."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO documents (user_id, filename, file_path, document_type, uploaded_at)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, user_id, filename, file_path, document_type, uploaded_at, processed, processed_at, content_preview
        """, (user_id, filename, file_path, document_type, datetime.utcnow()))
        return cursor.fetchone()

def get_document(document_id, user_id=None):
    """Get document by ID, optionally filtering by user_id."""
    with get_db_cursor() as cursor:
        if user_id:
            cursor.execute("""
            SELECT * FROM documents WHERE id = %s AND user_id = %s
            """, (document_id, user_id))
        else:
            cursor.execute("""
            SELECT * FROM documents WHERE id = %s
            """, (document_id,))
        return cursor.fetchone()

def get_documents_by_user(user_id, document_type=None):
    """Get all documents for a user, optionally filtering by type."""
    with get_db_cursor() as cursor:
        if document_type:
            cursor.execute("""
            SELECT * FROM documents WHERE user_id = %s AND document_type = %s
            """, (user_id, document_type))
        else:
            cursor.execute("""
            SELECT * FROM documents WHERE user_id = %s
            """, (user_id,))
        return cursor.fetchall()

def update_document_processed(document_id, output_path, content_preview=None):
    """Update a document's processed status."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE documents 
        SET processed = TRUE, processed_at = %s, output_path = %s, content_preview = %s
        WHERE id = %s
        RETURNING *
        """, (datetime.utcnow(), output_path, content_preview, document_id))
        return cursor.fetchone()

def update_document_error(document_id, error_message):
    """Update a document with processing error."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE documents 
        SET processing_error = %s
        WHERE id = %s
        RETURNING *
        """, (error_message, document_id))
        return cursor.fetchone()

def delete_document(document_id, user_id):
    """Delete a document by ID and user_id."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        DELETE FROM documents
        WHERE id = %s AND user_id = %s
        RETURNING *
        """, (document_id, user_id))
        return cursor.fetchone()

# Generation operations
def create_generation(user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory):
    """Create a new generation record."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO generations 
        (user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory, status, started_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """, (user_id, knowledge_base_id, agent_persona_id, questions_per_client, output_directory, 'pending', datetime.utcnow()))
        return cursor.fetchone()

def update_generation_status(generation_id, status, error_message=None):
    """Update a generation's status."""
    with get_db_cursor() as cursor:
        if status == 'completed':
            cursor.execute("""
            UPDATE generations 
            SET status = %s, completed_at = %s
            WHERE id = %s
            RETURNING *
            """, (status, datetime.utcnow(), generation_id))
        elif status == 'failed':
            cursor.execute("""
            UPDATE generations 
            SET status = %s, error_message = %s
            WHERE id = %s
            RETURNING *
            """, (status, error_message, generation_id))
        else:
            cursor.execute("""
            UPDATE generations 
            SET status = %s
            WHERE id = %s
            RETURNING *
            """, (status, generation_id))
        return cursor.fetchone()

def update_generation_counts(generation_id, client_types_count, questions_count):
    """Update a generation's counts."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET client_types_count = %s, questions_count = %s
        WHERE id = %s
        RETURNING *
        """, (client_types_count, questions_count, generation_id))
        return cursor.fetchone()

def get_generation(generation_id, user_id=None):
    """Get generation by ID, optionally filtering by user_id."""
    with get_db_cursor() as cursor:
        if user_id:
            cursor.execute("""
            SELECT * FROM generations WHERE id = %s AND user_id = %s
            """, (generation_id, user_id))
        else:
            cursor.execute("""
            SELECT * FROM generations WHERE id = %s
            """, (generation_id,))
        return cursor.fetchone()

def get_generations_by_user(user_id):
    """Get all generations for a user."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM generations WHERE user_id = %s
        """, (user_id,))
        return cursor.fetchall()

def delete_generation(generation_id, user_id):
    """Delete a generation by ID and user_id."""
    with get_db_cursor() as cursor:
        # First delete related client types
        cursor.execute("""
        DELETE FROM client_types
        WHERE generation_id = %s
        """, (generation_id,))
        
        # Then delete the generation
        cursor.execute("""
        DELETE FROM generations
        WHERE id = %s AND user_id = %s
        RETURNING *
        """, (generation_id, user_id))
        return cursor.fetchone()

# Client type operations
def create_client_type(generation_id, name, description, question_count, output_file):
    """Create a new client type record."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        INSERT INTO client_types 
        (generation_id, name, description, question_count, output_file)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING *
        """, (generation_id, name, description, question_count, output_file))
        return cursor.fetchone()

def get_client_types_by_generation(generation_id):
    """Get all client types for a generation."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        SELECT * FROM client_types WHERE generation_id = %s
        """, (generation_id,))
        return cursor.fetchall()

# Analysis operations
def update_generation_analysis(generation_id, analysis_path):
    """Update a generation with analysis information."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_path = %s
        WHERE id = %s
        RETURNING *
        """, (analysis_path, generation_id))
        return cursor.fetchone()

def complete_generation_analysis(generation_id):
    """Mark a generation's analysis as completed."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_completed = TRUE, analysis_completed_at = %s
        WHERE id = %s
        RETURNING *
        """, (datetime.utcnow(), generation_id))
        return cursor.fetchone()

def update_generation_analysis_error(generation_id, error_message):
    """Update a generation with analysis error."""
    with get_db_cursor() as cursor:
        cursor.execute("""
        UPDATE generations 
        SET analysis_error = %s
        WHERE id = %s
        RETURNING *
        """, (error_message, generation_id))
        return cursor.fetchone()

if __name__ == "__main__":
    # When run directly, initialize the database
    init_db()
    print("Database tables created.") 