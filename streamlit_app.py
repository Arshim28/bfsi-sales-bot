import os
import json
import time
import requests
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# API URL from environment or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page configuration
st.set_page_config(
    page_title="BFSI Sales Bot Generator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"


# Helper functions for API requests
def api_request(endpoint, method="GET", data=None, files=None, headers=None):
    """Make an API request with authentication."""
    if headers is None:
        headers = {}
    
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, files=files, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Check for successful response
        response.raise_for_status()
        
        # If response is JSON, return as data
        if response.headers.get("content-type") == "application/json":
            return response.json()
        # Otherwise return raw content
        return response.content
    
    except requests.exceptions.RequestException as e:
        error_msg = f"API Error: {str(e)}"
        try:
            error_data = e.response.json()
            if "detail" in error_data:
                error_msg = f"API Error: {error_data['detail']}"
        except:
            pass
        
        st.error(error_msg)
        return None


# Authentication functions
def login(username, password):
    """Log in user and set token."""
    data = {
        "username": username,
        "password": password
    }
    
    # Use form-encoded data for token endpoint
    try:
        response = requests.post(
            f"{API_URL}/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        token_data = response.json()
        
        # Store token in session
        st.session_state.token = token_data["access_token"]
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.current_page = "dashboard"
        
        # Give a moment for session state to update
        time.sleep(0.5)
        
        # Force a rerun to apply session state changes
        st.rerun()
        
        return True
    except requests.exceptions.RequestException as e:
        error_msg = "Login failed. Please check your credentials."
        try:
            error_data = e.response.json()
            if "detail" in error_data:
                error_msg = f"Login failed: {error_data['detail']}"
        except:
            pass
        
        st.error(error_msg)
        return False


def register(username, email, password):
    """Register a new user."""
    data = {
        "username": username,
        "email": email,
        "password": password
    }
    
    try:
        response = api_request("/users", method="POST", data=data)
        if response:
            st.success("Registration successful! Please log in.")
            st.session_state.current_page = "login"
            st.rerun()
            return True
        return False
    except:
        st.error("Registration failed. Please try again.")
        return False


def logout():
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.current_page = "login"
    
    st.rerun()


def navigate_to(page):
    st.session_state.current_page = page


# UI Components
def show_login_page():
    """Show the login page."""
    st.title("BFSI Sales Bot Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        with st.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Login")
            
            if submit_button:
                if username and password:
                    success = login(username, password)
                    if success:
                        # This is now handled by the login function with st.rerun()
                        pass
                else:
                    st.error("Please enter both username and password.")
    
    with col2:
        st.subheader("Register")
        with st.form(key="register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_button = st.form_submit_button(label="Register")
            
            if submit_button:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill out all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    register(new_username, new_email, new_password)
                    # Register function now handles the rerun


def show_sidebar():
    """Show the sidebar navigation."""
    with st.sidebar:
        st.title("Navigation")
        
        # User info
        st.write(f"Logged in as: **{st.session_state.username}**")
        
        # Navigation buttons
        st.button("Dashboard", on_click=navigate_to, args=("dashboard",))
        st.button("Documents", on_click=navigate_to, args=("documents",))
        st.button("Generations", on_click=navigate_to, args=("generations",))
        st.button("Analyses", on_click=navigate_to, args=("analyses",))
        
        # Logout button at the bottom
        st.sidebar.button("Logout", on_click=logout)


def show_dashboard():
    """Show the main dashboard."""
    st.title("Dashboard")
    
    # Fetch user's data
    documents = api_request("/documents") or []
    generations = api_request("/generations") or []
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", len(documents))
    
    with col2:
        st.metric("Generations", len(generations))
    
    with col3:
        completed_gens = sum(1 for g in generations if g.get("status") == "completed")
        st.metric("Completed Generations", completed_gens)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Upload New Document"):
            navigate_to("documents")
    
    with col2:
        if st.button("Create New Generation"):
            navigate_to("generations")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if generations:
        # Sort generations by started_at (recent first)
        recent_generations = sorted(
            generations, 
            key=lambda x: datetime.fromisoformat(x["started_at"].replace("Z", "+00:00")), 
            reverse=True
        )[:5]
        
        # Display recent generations
        for gen in recent_generations:
            with st.expander(f"Generation #{gen['id']} - {gen['status'].title()}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Status: **{gen['status'].title()}**")
                    st.write(f"Started: {gen['started_at']}")
                    if gen.get("completed_at"):
                        st.write(f"Completed: {gen['completed_at']}")
                
                with col2:
                    st.write(f"Client Types: {gen['client_types_count']}")
                    st.write(f"Questions: {gen['questions_count']}")
                    
                # View button
                st.button("View Details", key=f"view_gen_{gen['id']}", 
                          on_click=lambda id=gen['id']: (
                              setattr(st.session_state, "selected_generation", id),
                              navigate_to("generation_details")
                          ))
    else:
        st.info("No generations yet. Create one from the Generations page.")


def show_documents_page():
    """Show the documents page."""
    st.title("Documents")
    
    # Upload section
    with st.expander("Upload New Document", expanded=True):
        with st.form("upload_form"):
            document_type = st.selectbox(
                "Document Type",
                options=["knowledge_base", "agent_persona"],
                format_func=lambda x: "Knowledge Base" if x == "knowledge_base" else "Agent Persona"
            )
            uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
            
            submit_button = st.form_submit_button("Upload")
            if submit_button and uploaded_file:
                # Create a temporary file to store the upload
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Prepare form data for upload
                files = {
                    "file": (uploaded_file.name, open(tmp_path, "rb"), "text/plain")
                }
                
                # Make API request
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files=files,
                    data={"document_type": document_type},
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                # Clean up the temporary file
                os.unlink(tmp_path)
                
                # Handle the response
                if response.status_code == 200:
                    st.success(f"Document uploaded successfully: {uploaded_file.name}")
                    # Refresh the page to see the new document
                    st.rerun()
                else:
                    try:
                        error = response.json()
                        st.error(f"Upload failed: {error.get('detail', 'Unknown error')}")
                    except:
                        st.error(f"Upload failed: {response.text}")
    
    # Fetch and display documents
    documents = api_request("/documents") or []
    
    if documents:
        # Separate documents by type
        kb_docs = [d for d in documents if d["document_type"] == "knowledge_base"]
        persona_docs = [d for d in documents if d["document_type"] == "agent_persona"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Knowledge Base Documents")
            if kb_docs:
                for doc in kb_docs:
                    with st.expander(f"{doc['filename']} (ID: {doc['id']})"):
                        st.write(f"Uploaded: {doc['uploaded_at']}")
                        st.write(f"Processed: {'✅' if doc['processed'] else '⏳'}")
                        
                        if doc['processed'] and doc.get('content_preview'):
                            st.subheader("Preview")
                            st.text(doc['content_preview'])
                        
                        # Delete button
                        if st.button("Delete", key=f"del_kb_{doc['id']}"):
                            if api_request(f"/documents/{doc['id']}", method="DELETE"):
                                st.success("Document deleted successfully.")
                                # Refresh the page
                                st.rerun()
            else:
                st.info("No knowledge base documents uploaded yet.")
        
        with col2:
            st.subheader("Agent Persona Documents")
            if persona_docs:
                for doc in persona_docs:
                    with st.expander(f"{doc['filename']} (ID: {doc['id']})"):
                        st.write(f"Uploaded: {doc['uploaded_at']}")
                        st.write(f"Processed: {'✅' if doc['processed'] else '⏳'}")
                        
                        if doc['processed'] and doc.get('content_preview'):
                            st.subheader("Preview")
                            st.text(doc['content_preview'])
                        
                        # Delete button
                        if st.button("Delete", key=f"del_ap_{doc['id']}"):
                            if api_request(f"/documents/{doc['id']}", method="DELETE"):
                                st.success("Document deleted successfully.")
                                # Refresh the page
                                st.rerun()
            else:
                st.info("No agent persona documents uploaded yet.")
    else:
        st.info("No documents uploaded yet. Use the form above to upload your first document.")


def show_generations_page():
    """Show the generations page."""
    st.title("Prompt Generations")
    
    # Get documents for generation
    documents = api_request("/documents") or []
    kb_docs = [d for d in documents if d["document_type"] == "knowledge_base" and d["processed"]]
    persona_docs = [d for d in documents if d["document_type"] == "agent_persona" and d["processed"]]
    
    # Create new generation section
    with st.expander("Create New Generation", expanded=True):
        if kb_docs and persona_docs:
            with st.form("generation_form"):
                kb_id = st.selectbox(
                    "Knowledge Base Document",
                    options=[doc["id"] for doc in kb_docs],
                    format_func=lambda x: next((d["filename"] for d in kb_docs if d["id"] == x), "Unknown")
                )
                
                persona_id = st.selectbox(
                    "Agent Persona Document",
                    options=[doc["id"] for doc in persona_docs],
                    format_func=lambda x: next((d["filename"] for d in persona_docs if d["id"] == x), "Unknown")
                )
                
                questions_per_client = st.slider(
                    "Questions per Client Type",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5
                )
                
                submit_button = st.form_submit_button("Generate Prompts")
                if submit_button:
                    data = {
                        "knowledge_base_id": kb_id,
                        "agent_persona_id": persona_id,
                        "questions_per_client": questions_per_client
                    }
                    
                    response = api_request("/generations", method="POST", data=data)
                    if response:
                        st.success("Generation started successfully!")
                        # Set a flag to start polling for generation status
                        st.session_state.polling_generation = response["id"]
                        st.session_state.selected_generation = response["id"]
                        navigate_to("generation_details")
        else:
            st.warning("You need at least one processed knowledge base document and one agent persona document to create a generation.")
            st.button("Upload Documents", on_click=navigate_to, args=("documents",))
    
    # Fetch and display generations
    generations = api_request("/generations") or []
    
    if generations:
        # Display generations in a table
        gen_data = []
        for gen in generations:
            gen_data.append({
                "ID": gen["id"],
                "Status": gen["status"].capitalize(),
                "Started": gen["started_at"],
                "Client Types": gen["client_types_count"],
                "Questions": gen["questions_count"],
                "Analyzed": "✅" if gen["analysis_completed"] else "❌"
            })
        
        df = pd.DataFrame(gen_data)
        st.dataframe(df, use_container_width=True)
        
        # View generation details
        col1, col2 = st.columns(2)
        with col1:
            selected_id = st.selectbox(
                "Select Generation to View",
                options=[gen["id"] for gen in generations],
                format_func=lambda x: f"Generation #{x}"
            )
        
        with col2:
            if selected_id:
                if st.button("View Generation Details"):
                    st.session_state.selected_generation = selected_id
                    navigate_to("generation_details")
    else:
        st.info("No generations created yet.")


def show_generation_details():
    """Show details for a specific generation."""
    generation_id = st.session_state.get("selected_generation")
    
    if not generation_id:
        st.error("No generation selected.")
        st.button("Back to Generations", on_click=navigate_to, args=("generations",))
        return
    
    # Fetch generation details
    generation = api_request(f"/generations/{generation_id}")
    if not generation:
        st.error("Generation not found.")
        st.button("Back to Generations", on_click=navigate_to, args=("generations",))
        return
    
    # Display generation info
    st.title(f"Generation #{generation['id']}")
    
    # Status and controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = {
            "pending": "blue",
            "processing": "orange",
            "completed": "green",
            "failed": "red"
        }.get(generation["status"], "gray")
        
        st.markdown(f"Status: :{status_color}[**{generation['status'].upper()}**]")
    
    with col2:
        st.write(f"Started: {generation['started_at']}")
        if generation.get("completed_at"):
            st.write(f"Completed: {generation['completed_at']}")
    
    with col3:
        # Buttons based on generation status
        if generation["status"] == "completed" and not generation.get("analysis_completed"):
            if st.button("Run Analysis"):
                # Send analysis request
                response = api_request(
                    "/analysis",
                    method="POST",
                    data={"generation_id": generation["id"]}
                )
                if response:
                    st.success("Analysis started!")
                    time.sleep(1)  # Small delay
                    st.rerun()
        
        elif generation["status"] == "completed" and generation.get("analysis_completed"):
            if st.button("View Analysis"):
                # Get the analysis data
                try:
                    analysis_data = requests.get(
                        f"{API_URL}/analysis/{generation['id']}",
                        headers={"Authorization": f"Bearer {st.session_state.token}"}
                    ).content.decode("utf-8")
                    
                    # Store in session and navigate
                    st.session_state.analysis_data = analysis_data
                    st.session_state.analysis_generation_id = generation["id"]
                    navigate_to("analysis_view")
                except Exception as e:
                    st.error(f"Failed to fetch analysis: {str(e)}")
    
    # Delete button
    if st.button("Delete Generation"):
        if api_request(f"/generations/{generation['id']}", method="DELETE"):
            st.success("Generation deleted successfully.")
            navigate_to("generations")
            return
    
    # Display client types and questions
    if generation.get("client_types"):
        st.subheader(f"Client Types ({len(generation['client_types'])})")
        
        for client_type in generation["client_types"]:
            with st.expander(f"{client_type['name']} - {client_type['question_count']} questions"):
                st.markdown(f"**Description**: {client_type['description']}")
                st.write(f"Created: {client_type['created_at']}")
                
                # If we have the output file path, we could also display sample questions
                if client_type.get("output_file") and os.path.exists(client_type["output_file"]):
                    st.subheader("Sample Questions")
                    try:
                        with open(client_type["output_file"], "r") as f:
                            data = json.load(f)
                            if "questions" in data and data["questions"]:
                                # Display just a few sample questions
                                for i, q in enumerate(data["questions"][:5]):
                                    st.write(f"{i+1}. {q['text']}")
                                
                                if len(data["questions"]) > 5:
                                    st.write("...")
                    except:
                        st.error("Failed to load questions from file.")
    
    # If the generation is still processing, set up automatic refresh
    if generation["status"] in ["pending", "processing"]:
        st.info("This generation is still processing. The page will refresh automatically.")
        # Add a refresh button
        if st.button("Refresh Now"):
            st.rerun()
        # Auto-refresh using JavaScript
        st.markdown(
            """
            <script>
                setTimeout(function() {
                    window.location.reload();
                }, 5000);  // Refresh every 5 seconds
            </script>
            """,
            unsafe_allow_html=True
        )


def show_analyses_page():
    """Show the analyses page."""
    st.title("Analyses")
    
    # Fetch completed generations with analyses
    generations = api_request("/generations") or []
    analyzed_generations = [g for g in generations if g.get("analysis_completed")]
    
    if analyzed_generations:
        # Display available analyses
        for gen in analyzed_generations:
            with st.expander(f"Analysis for Generation #{gen['id']}"):
                st.write(f"Completed: {gen.get('analysis_completed_at')}")
                st.write(f"Client Types: {gen['client_types_count']}")
                st.write(f"Total Questions: {gen['questions_count']}")
                
                # View button
                if st.button("View Analysis", key=f"view_analysis_{gen['id']}"):
                    try:
                        analysis_data = requests.get(
                            f"{API_URL}/analysis/{gen['id']}",
                            headers={"Authorization": f"Bearer {st.session_state.token}"}
                        ).content.decode("utf-8")
                        
                        # Store in session and navigate
                        st.session_state.analysis_data = analysis_data
                        st.session_state.analysis_generation_id = gen["id"]
                        navigate_to("analysis_view")
                    except Exception as e:
                        st.error(f"Failed to fetch analysis: {str(e)}")
    else:
        st.info("No analyses available yet. Complete a generation and run analysis on it.")
        # Button to go to generations
        st.button("Go to Generations", on_click=navigate_to, args=("generations",))


def show_analysis_view():
    """Show a specific analysis."""
    analysis_data = st.session_state.get("analysis_data")
    generation_id = st.session_state.get("analysis_generation_id")
    
    if not analysis_data or not generation_id:
        st.error("Analysis data not found.")
        st.button("Back to Analyses", on_click=navigate_to, args=("analyses",))
        return
    
    st.title(f"Analysis for Generation #{generation_id}")
    
    # Back button
    if st.button("← Back to Analyses"):
        navigate_to("analyses")
    
    # Render markdown
    st.markdown(analysis_data)


# Main app logic
def main():
    """Main app function."""
    # Check if user is authenticated
    if not st.session_state.authenticated:
        # Debug info - can be removed later
        st.sidebar.info(f"Session state: not authenticated")
        show_login_page()
    else:
        # Debug info - can be removed later
        st.sidebar.success(f"Authenticated as: {st.session_state.username}")
        
        # Show sidebar
        show_sidebar()
        
        # Show the selected page
        if st.session_state.current_page == "dashboard":
            show_dashboard()
        elif st.session_state.current_page == "documents":
            show_documents_page()
        elif st.session_state.current_page == "generations":
            show_generations_page()
        elif st.session_state.current_page == "generation_details":
            show_generation_details()
        elif st.session_state.current_page == "analyses":
            show_analyses_page()
        elif st.session_state.current_page == "analysis_view":
            show_analysis_view()
        else:
            # Default to dashboard
            show_dashboard()


if __name__ == "__main__":
    main() 