import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .utils import setup_logger


load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = setup_logger(__name__, level="DEBUG")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
class ClientType:
    def __init__(self, client_type, description):
        self.client_type = client_type
        self.description = description
        
    def dict(self):
        return {"client_type": self.client_type, "description": self.description}

class Question:
    def __init__(self, question, context=""):
        self.question = question
        self.context = context
        
    def dict(self):
        return {"question": self.question, "context": self.context}

class Response:
    def __init__(self, question="", response="", key_points=None):
        self.question = question
        self.response = response
        self.key_points = key_points or []
        
    def dict(self):
        return {"question": self.question, "response": self.response, "key_points": self.key_points}

def read_content_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"Successfully read content from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to read content from {file_path}: {e}", exc_info=True)
        raise

def init_llm(api_key: str) -> genai.Client:
    client = genai.Client(api_key=api_key)
    return client

def generate_response(
    question: str,
    knowledge_base: str, 
    persona: str, 
    model: str,
    client_type: ClientType
) -> Response:
    try:
        client = init_llm(GOOGLE_API_KEY)
        
        if not knowledge_base or not persona or not client_type.description or not question:
            logger.warning(f"One or more critical inputs to generate_response for client '{client_type.client_type}' are empty.")
            
        logger.debug(f"Preparing to generate response for client '{client_type.client_type}', question '{question[:50]}...'")
        
        prompt = f"""
        You are a financial services sales assistant responding to a client question.
        Use the knowledge base to provide accurate information and follow the agent persona 
        for tone and style. Tailor your response to the specific client type.
        
        Knowledge Base:
        {knowledge_base}
        
        Agent Persona:
        {persona}
        
        Client Type:
        {client_type.description}
        
        Question:
        {question}
        
        Provide a detailed, helpful response and list 3-5 key points covered in your response.
        Format your response as valid JSON with:
        {{"response": "Your detailed response", "key_points": ["point 1", "point 2", ...]}}
        """
        
        generation_response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192
            )
        )
        
        result = generation_response.text.strip()
        
        try:
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error(f"Failed to extract JSON from response: {result}")
                return Response(
                    question=question,
                    response=result,
                    key_points=["Response was not properly formatted"]
                )
            
            json_str = result[start_idx:end_idx]
            response_data = json.loads(json_str)
            
            return Response(
                question=question,
                response=response_data.get("response", ""),
                key_points=response_data.get("key_points", [])
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}, content: {result}")
            return Response(
                question=question,
                response="I apologize, but I'm unable to process your request at this time. Please contact our customer service for assistance.",
                key_points=["Error parsing response", "Redirected to customer service"]
            )
    except Exception as e:
        logger.error(f"Failed to generate response for question '{question}': {e}", exc_info=True)
        return Response(
            question=question,
            response=f"I apologize, but I'm unable to provide a specific answer to this question at this time. Please contact our customer service for more detailed information.",
            key_points=["Error generating response", "Redirected to customer service"]
        )

def generate_questions(
    knowledge_base: str, 
    persona: str, 
    client_type: ClientType,
    model: str,
    num_questions: int = 5
) -> List[Question]:
    try:
        client = init_llm(GOOGLE_API_KEY)
        
        logger.debug(f"Preparing to generate {num_questions} questions for client '{client_type.client_type}'")
        
        prompt = f"""
        You are a financial services sales assistant. Generate exactly {num_questions} specific questions that the following client type might ask.
        Create questions that are relevant to the knowledge base and can be answered by the agent persona.
        
        Knowledge Base:
        {knowledge_base}
        
        Agent Persona:
        {persona}
        
        Client Type:
        {client_type.description}
        
        For each question, provide:
        1. A specific, realistic question this client type would ask (must be a complete question ending with a question mark)
        2. Context about why this client would ask this question (2-3 sentences)
        
        Format your response as valid JSON with an array of objects containing "question" and "context" fields.
        Ensure you generate exactly {num_questions} questions, no more and no less.
        
        Example:
        [
            {{
                "question": "What investment options do you offer for someone looking to save for retirement?",
                "context": "This client is worried about long-term financial security. They want to understand their options for building a retirement fund with regular contributions."
            }},
            ...
        ]
        """
        
        generation_response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=4096
            )
        )
        
        result = generation_response.text.strip()
        start_idx = result.find('[')
        end_idx = result.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            logger.error(f"Failed to extract JSON from response: {result}")
            result = f'[{result}]'
        else:
            result = result[start_idx:end_idx]
            
        questions_data = json.loads(result)
        
        # Ensure we have exactly the requested number of questions
        questions_data = questions_data[:num_questions]
        
        # If we don't have enough questions, add generic ones
        if len(questions_data) < num_questions:
            logger.warning(f"Only generated {len(questions_data)} questions, adding generic ones to reach {num_questions}")
            generic_questions = [
                {"question": f"What services do you offer for {client_type.client_type}?", 
                 "context": f"The {client_type.client_type} needs basic information about available services."},
                {"question": "How secure is my data with your financial institution?", 
                 "context": "Security is a common concern for all financial service users."},
                {"question": "What fees are associated with your services?", 
                 "context": "Cost is an important factor in financial decision-making."},
                {"question": "How can I contact customer support if I need help?", 
                 "context": "Access to support is essential for resolving issues."},
                {"question": "What makes your financial services better than competitors?", 
                 "context": "Comparing options is a standard practice before committing to a financial service."}
            ]
            
            needed = num_questions - len(questions_data)
            questions_data.extend(generic_questions[:needed])
        
        questions = [Question(q.get("question", ""), q.get("context", "")) for q in questions_data]
        
        logger.info(f"Generated {len(questions)} questions for client type: {client_type.client_type}")
        return questions
    except Exception as e:
        logger.error(f"Failed to generate questions for {client_type.client_type}: {e}", exc_info=True)
        return []

def process_client_type(
    knowledge_base: str,
    persona: str,
    client_type: ClientType,
    model: str,
    output_dir: Path,
    questions_per_client: int = 10
) -> bool:
    try:
        client_dir = output_dir / client_type.client_type
        os.makedirs(client_dir, exist_ok=True)
        
        with open(client_dir / "client_type.json", "w") as f:
            json.dump({"client_type": client_type.client_type, "description": client_type.description}, f, indent=2)
        
        logger.info(f"Generating questions for client type: {client_type.client_type}")
        
        questions = generate_questions(
            knowledge_base=knowledge_base,
            persona=persona,
            client_type=client_type,
            model=model,
            num_questions=questions_per_client
        )
        
        if not questions:
            logger.error(f"Failed to generate questions for client type: {client_type.client_type}")
            return False
        
        logger.info(f"Generated {len(questions)} questions for client type: {client_type.client_type}")
        
        with open(client_dir / "questions.json", "w") as f:
            json.dump([q.dict() for q in questions], f, indent=2)
        
        logger.info(f"Generating responses for client type: {client_type.client_type}")
        
        responses = []
        for i, question in enumerate(questions):
            logger.info(f"Generating response for question {i+1}/{len(questions)}")
            
            response = generate_response(
                question=question.question,
                knowledge_base=knowledge_base,
                persona=persona,
                client_type=client_type,
                model=model
            )
            
            if response:
                responses.append(Response(question=question.question, response=response.response, key_points=response.key_points))
            else:
                logger.warning(f"Failed to generate response for question: {question.question}")
        
        logger.info(f"Generated {len(responses)} responses for client type: {client_type.client_type}")
        
        with open(client_dir / "responses.json", "w") as f:
            json.dump([r.dict() for r in responses], f, indent=2)
        
        # Also create a text prompt file in the format expected by the analyzer
        with open(output_dir / f"{client_type.client_type}_prompt.txt", "w") as f:
            f.write(f"# Client Type: {client_type.client_type}\n\n")
            f.write(f"{client_type.description}\n\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (question, response) in enumerate(zip(questions, responses), 1):
                f.write(f"## Q{i}: {question.question}\n\n")
                f.write(f"Context: {question.context}\n\n")
                f.write(f"Response: {response.response}\n\n")
                f.write("Key points:\n")
                for point in response.key_points:
                    f.write(f"- {point}\n")
                f.write("\n" + "-" * 80 + "\n\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to process client type {client_type.client_type}: {e}", exc_info=True)
        return False

def format_final_outputs(output_dir: Path, username: str):
    """
    Format the final required outputs.
    
    Args:
        output_dir: Directory containing generation outputs.
        username: Username for the generation.
        
    Returns:
        Tuple of (persona_path, kb_qa_path, analysis_path)
    """
    try:
        logger.info(f"Formatting final outputs for {username} from {output_dir}")
        
        # 1. Create bot persona file
        persona_path = output_dir.parent / f"{username}_server_bot_persona.md"
        
        # Combine all client types into a comprehensive persona
        client_type_data = []
        for client_dir in [d for d in output_dir.iterdir() if d.is_dir()]:
            client_type_file = client_dir / "client_type.json"
            if client_type_file.exists():
                with open(client_type_file, 'r') as f:
                    client_type_data.append(json.load(f))
        
        with open(persona_path, 'w') as f:
            f.write("# BFSI Sales Bot Persona\n\n")
            f.write("## Overall Persona\n\n")
            f.write("The BFSI Sales Bot is an AI-powered financial services assistant designed to provide accurate, helpful information about banking, financial services, and insurance products. The bot maintains a professional, empathetic tone while adapting its communication style to different client types. It focuses on understanding client needs, providing tailored solutions, and guiding clients through their financial journey.\n\n")
            f.write("## Client Types\n\n")
            
            for client in client_type_data:
                f.write(f"### {client.get('client_type', 'Unknown')}\n\n")
                f.write(f"{client.get('description', '')}\n\n")
        
        # 2. Create knowledge base Q&A file
        kb_qa_path = output_dir.parent / f"{username}_server_bot_knowledge_base.md"
        
        with open(kb_qa_path, 'w') as f:
            f.write("# BFSI Sales Bot Knowledge Base (Q&A Format)\n\n")
            
            # Gather all Q&A pairs from all client types
            for client_dir in [d for d in output_dir.iterdir() if d.is_dir()]:
                responses_file = client_dir / "responses.json"
                client_type_file = client_dir / "client_type.json"
                
                if not all(f.exists() for f in [responses_file, client_type_file]):
                    continue
                    
                # Get client type name
                with open(client_type_file, 'r') as cf:
                    client_data = json.load(cf)
                    client_name = client_data.get('client_type', client_dir.name)
                
                f.write(f"## {client_name} Questions\n\n")
                
                # Read responses
                with open(responses_file, 'r') as rf:
                    responses = json.load(rf)
                    
                    for i, response in enumerate(responses, 1):
                        f.write(f"### Q{i}: {response.get('question', '')}\n\n")
                        f.write(f"Answer: {response.get('response', '')}\n\n")
                        f.write("Key points:\n")
                        for point in response.get('key_points', []):
                            f.write(f"- {point}\n")
                        f.write("\n---\n\n")
        
        # 3. Analysis report is already created by analyze_prompts()
        # Find the most recent analysis file
        analysis_dir = output_dir.parent.parent / "analysis"
        analysis_files = list(analysis_dir.glob(f"{username}_analysis_*.md"))
        analysis_path = None
        
        if analysis_files:
            analysis_path = max(analysis_files, key=lambda p: p.stat().st_mtime)
            
        logger.info(f"Final outputs created: Persona: {persona_path}, KB: {kb_qa_path}, Analysis: {analysis_path}")
        return (persona_path, kb_qa_path, analysis_path)
    except Exception as e:
        logger.error(f"Failed to format final outputs: {e}", exc_info=True)
        raise

def create_prompts(
    knowledge_base_path: str, 
    agent_persona_path: str, 
    questions_per_client: int = 5,
    model: str = "gemini-2.0-flash",
    output_dir: Path = None,
    username_for_logging: str = "api_user"
):
    try:
        logger.info(f"Starting prompt creation for user: {username_for_logging}")
        
        base_dir = Path(__file__).resolve().parent.parent
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_dir / "prompts" / f"{username_for_logging}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not Path(knowledge_base_path).exists():
            raise FileNotFoundError(f"Knowledge base file not found: {knowledge_base_path}")
        if not Path(agent_persona_path).exists():
            raise FileNotFoundError(f"Agent persona file not found: {agent_persona_path}")
        
        knowledge_base = read_content_file(knowledge_base_path)
        persona = read_content_file(agent_persona_path)
        
        client = init_llm(GOOGLE_API_KEY)
        
        prompt = f"""
        Based on the following knowledge base and agent persona information, identify exactly 2 distinct 
        client types that would use these financial services. For each client type, provide:
        1. A short identifier (2-3 words with underscores, e.g., "rookie_trader")
        2. A detailed description of this client type (150-200 words)
        
        Knowledge Base:
        {knowledge_base}
        
        Agent Persona:
        {persona}
        
        Return your answer in valid JSON format with an array of objects containing "client_type" and "description" fields:
        [
            {{
                "client_type": "client_type_identifier",
                "description": "Detailed description of this client type"
            }},
            {{
                "client_type": "second_client_type",
                "description": "Detailed description of second client type"
            }}
        ]
        """
        
        try:
            logger.info("Generating client types...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096
                )
            )
            
            result = response.text.strip()
            start_idx = result.find('[')
            end_idx = result.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error(f"Failed to extract JSON from response: {result}")
                result = f'[{result}]'
            else:
                result = result[start_idx:end_idx]
            
            client_types_data = json.loads(result)
            # Ensure we only have 2 client types
            client_types_data = client_types_data[:2]
            client_types_list = [ClientType(data.get("client_type", ""), data.get("description", "")) for data in client_types_data]
            
            if not client_types_list:
                logger.error("No client types were generated")
                return []
                
            logger.info(f"Successfully generated {len(client_types_list)} client types")
        except Exception as e:
            logger.error(f"Failed to generate client types: {e}", exc_info=True)
            return []
        
        results = []
        for client_type_obj in client_types_list:
            if not hasattr(client_type_obj, 'client_type') or not hasattr(client_type_obj, 'description'):
                logger.error(f"Invalid object in client_types_list: {client_type_obj}")
                continue

            success = process_client_type(
                knowledge_base, persona, client_type_obj, model, output_dir, questions_per_client
            )
            results.append((client_type_obj.client_type, success))
            
            # Add questions attribute to client_type_obj for compatibility with the API
            questions_file = output_dir / client_type_obj.client_type / "questions.json"
            if questions_file.exists():
                with open(questions_file, 'r') as f:
                    questions_data = json.load(f)
                    setattr(client_type_obj, 'questions', questions_data)
            else:
                setattr(client_type_obj, 'questions', [])
        
        logger.info("Prompt creation completed")
        for client_name, success_status in results:
            status_msg = "Success" if success_status else "Failed"
            logger.info(f"Client type '{client_name}': {status_msg}")
        
        # Generate final output files
        format_final_outputs(output_dir, username_for_logging)
        
        return client_types_list
    except Exception as e:
        logger.error(f"Failed to create prompts: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create prompts for a BFSI sales bot.")
    parser.add_argument("--kb_path", type=str, required=True, help="Path to the knowledge base markdown file.")
    parser.add_argument("--persona_path", type=str, required=True, help="Path to the agent persona markdown file.")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions per client type.")
    parser.add_argument("--output_dir", type=str, help="Directory to save prompts to.")
    parser.add_argument("--log_user", type=str, default="script_user", help="Username for logging.")

    args = parser.parse_args()
    
    output_dir_path = Path(args.output_dir) if args.output_dir else None
    
    generated_client_types = create_prompts(
        args.kb_path, 
        args.persona_path, 
        args.questions, 
        model="gemini-2.0-flash",
        output_dir=output_dir_path,
        username_for_logging=args.log_user
    )
    
    if not generated_client_types:
        logger.error("Failed to create prompts - see logs for details")
        exit(1)
    
    logger.info("Successfully created prompts")