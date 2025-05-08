# BFSI Sales Bot Generator

A powerful tool for generating financial sales bots based on knowledge bases and agent personas. This project allows financial institutions to create customized sales chatbots with minimal effort.

## Features

- Upload and parse knowledge base and agent persona documents
- Generate distinct client types and targeted questions with responses
- Analyze prompt quality and get improvement suggestions
- RESTful API for integration with other systems
- Streamlit user interface for easy interaction
- User authentication with JWT and API keys
- PostgreSQL database for persistent storage

## Project Structure

```
├── app.py              # Main FastAPI application entrypoint
├── streamlit_app.py    # Streamlit frontend application
├── run_streamlit.py    # Helper script to run the Streamlit app
├── bot/                # Core modules
│   ├── analyzer.py     # LLM-based prompt analysis
│   ├── api.py          # FastAPI routes and endpoints
│   ├── auth.py         # Authentication utilities
│   ├── creator.py      # Prompt generation logic
│   ├── database.py     # Database functions and connection
│   ├── parser.py       # Document parsing
│   ├── schemas.py      # Pydantic models for request/response
│   └── utils.py        # Shared utilities
├── data/               # Data storage
│   ├── analysis/       # Generated analysis reports
│   ├── prompts/        # Generated prompts
│   └── uploads/        # Uploaded documents
└── requirements.txt    # Project dependencies
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bfsi-bot-gen.git
   cd bfsi-bot-gen
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   DATABASE_URL=postgresql://username:password@localhost/bfsi_bot_gen
   SECRET_KEY=your_secret_key_for_jwt
   GOOGLE_API_KEY=your_google_gemini_api_key
   API_URL=http://localhost:8000
   ```

4. Set up PostgreSQL database:
   ```bash
   createdb bfsi_bot_gen  # Create PostgreSQL database
   ```

5. Run the backend API:
   ```bash
   python app.py
   ```

6. In a separate terminal, run the Streamlit frontend:
   ```bash
   python run_streamlit.py
   ```

## Using the Application

### Streamlit User Interface

The application provides a user-friendly Streamlit interface that can be accessed at http://localhost:8501 by default.

1. **Login/Register**: Create an account or login with existing credentials
2. **Dashboard**: View statistics and recent activity
3. **Documents**: Upload and manage knowledge base and agent persona documents
4. **Generations**: Create and monitor prompt generation tasks
5. **Analyses**: View detailed analyses of generated prompts

To run the Streamlit interface with custom options:
```bash
python run_streamlit.py --port 8502 --api-url http://custom-api-url:8000
```

### API Endpoints

The backend API is available at http://localhost:8000 by default and provides the following endpoints:

#### Authentication
- `POST /token` - Get JWT token with username/password
- `POST /users` - Create a new user

#### Documents
- `POST /documents/upload` - Upload a document (knowledge base or agent persona)
- `GET /documents` - List all user's documents
- `GET /documents/{document_id}` - Get document details
- `DELETE /documents/{document_id}` - Delete a document

#### Generations
- `POST /generations` - Start a new prompt generation
- `GET /generations` - List all user's generations
- `GET /generations/{generation_id}` - Get generation details
- `DELETE /generations/{generation_id}` - Delete a generation

#### Analysis
- `POST /analysis` - Start a new prompt analysis
- `GET /analysis/{generation_id}` - Get analysis report

## API Usage Example

1. Create a user account:
   ```bash
   curl -X POST "http://localhost:8000/users" \
     -H "Content-Type: application/json" \
     -d '{"username":"testuser","email":"test@example.com","password":"password123"}'
   ```

2. Get authentication token:
   ```bash
   curl -X POST "http://localhost:8000/token" \
     -F "username=testuser" \
     -F "password=password123"
   ```

3. Upload documents:
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "document_type=knowledge_base" \
     -F "file=@/path/to/knowledge_base.txt"
   ```

4. Generate prompts:
   ```bash
   curl -X POST "http://localhost:8000/generations" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"knowledge_base_id":1,"agent_persona_id":2,"questions_per_client":50}'
   ```

5. Analyze prompts:
   ```bash
   curl -X POST "http://localhost:8000/analysis" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"generation_id":1}'
   ```

## Advanced Configuration

### Environment Variables

- `PORT` - API server port (default: 8000)
- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - JWT token secret key
- `GOOGLE_API_KEY` - Google Gemini API key
- `API_URL` - URL where the FastAPI backend is running (for Streamlit frontend)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
