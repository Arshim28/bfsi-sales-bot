import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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
    num_questions: int = 10
) -> List[Question]:
    try:
        client = init_llm(GOOGLE_API_KEY)
        
        prompt = f"""
        You are a financial services analyst creating questions that a {client_type.description} client might ask.
        
        Knowledge Base:
        {knowledge_base}
        
        Agent Persona:
        {persona}
        
        Please generate {num_questions} realistic questions that this client type might ask about the financial services 
        described in the knowledge base. For each question, provide relevant context that would help an agent prepare a response.
        
        Return the questions in JSON format:
        [
            {{
                "question": "question text here",
                "context": "relevant context here that would help prepare a response"
            }},
            ...
        ]
        
        Each question should be specific and realistic. The context should provide relevant background information.
        """
        
        response = client.models.generate_content(
            model=model, 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192
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
        
        questions_data = json.loads(result)
        
        questions = []
        for q_data in questions_data:
            questions.append(Question(
                question=q_data.get("question", ""),
                context=q_data.get("context", "")
            ))
        
        logger.info(f"Generated {len(questions)} questions for client type: {client_type.client_type}")
        return questions
    except Exception as e:
        logger.error(f"Failed to generate questions for client type {client_type.client_type}: {e}", exc_info=True)
        if num_questions > 5:
            logger.info("Attempting to generate a smaller batch of questions...")
            return generate_questions(knowledge_base, persona, client_type, model, 5)
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
        
        return True
    except Exception as e:
        logger.error(f"Failed to process client type {client_type.client_type}: {e}", exc_info=True)
        return False

def create_prompts(
    knowledge_base_path: str, 
    agent_persona_path: str, 
    questions_per_client: int = 10,
    model: str = "gemini-2.0-flash",
    output_dir: Path = None,
    username_for_logging: str = "api_user"
):
    try:
        logger.info(f"Starting prompt creation for user: {username_for_logging}")
        
        base_dir = Path(__file__).resolve().parent.parent
        if output_dir is None:
            output_dir = base_dir / "prompts" / username_for_logging 
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not Path(knowledge_base_path).exists():
            raise FileNotFoundError(f"Knowledge base file not found: {knowledge_base_path}")
        if not Path(agent_persona_path).exists():
            raise FileNotFoundError(f"Agent persona file not found: {agent_persona_path}")
        
        knowledge_base = read_content_file(knowledge_base_path)
        persona = read_content_file(agent_persona_path)
        
        client = init_llm(GOOGLE_API_KEY)
        
        prompt = f"""
        Based on the following knowledge base and agent persona information, identify 5-7 distinct 
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
            ...
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
        
        logger.info("Prompt creation completed")
        for client_name, success_status in results:
            status_msg = "Success" if success_status else "Failed"
            logger.info(f"Client type '{client_name}': {status_msg}")
        
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
