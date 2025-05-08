import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the API with uvicorn
    uvicorn.run(
        "bot.api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
