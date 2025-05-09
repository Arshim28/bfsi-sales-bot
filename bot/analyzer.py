import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import glob
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .utils import setup_logger

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = setup_logger(__name__)

class ClientTypeAnalysis(BaseModel):
    client_type: str = Field(description="The client type being analyzed")
    description_quality: int = Field(description="Rating of description quality (1-10)")
    description_feedback: str = Field(description="Feedback on the client type description")
    question_quality: int = Field(description="Overall rating of question quality (1-10)")
    question_feedback: str = Field(description="Feedback on the questions generated")
    response_quality: int = Field(description="Overall rating of response quality (1-10)")
    response_feedback: str = Field(description="Feedback on the responses generated")
    improvement_suggestions: List[str] = Field(description="Suggestions for improvement")

class PromptSetAnalysis(BaseModel):
    username: str = Field(description="Username of the prompt set owner")
    overall_quality: int = Field(description="Overall quality rating (1-10)")
    client_type_analyses: List[ClientTypeAnalysis] = Field(description="Analysis of each client type")
    strengths: List[str] = Field(description="Overall strengths of the prompt set")
    weaknesses: List[str] = Field(description="Overall weaknesses of the prompt set")
    improvement_suggestions: List[str] = Field(description="Overall suggestions for improvement")
    summary: str = Field(description="Executive summary of the analysis")

def init_llm():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        client = genai.Client(api_key=api_key)
        logger.info("Successfully initialized Google Gemini client for analysis")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise

def read_client_type_data(client_dir: Path) -> Dict[str, Any]:
    try:
        client_type_file = client_dir / "client_type.json"
        questions_file = client_dir / "questions.json"
        responses_file = client_dir / "responses.json"
        
        if not all(f.exists() for f in [client_type_file, questions_file, responses_file]):
            logger.warning(f"Missing files in {client_dir}")
            return None
        
        # Read client type data
        with open(client_type_file, 'r', encoding='utf-8') as f:
            client_data = json.load(f)
        
        # Read questions data
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Read responses data
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)
        
        # Create prompt data structure
        prompt_data = {
            'client_type': client_data.get('client_type', client_dir.name),
            'description': client_data.get('description', ''),
            'qa_pairs': []
        }
        
        # Match questions with responses
        for response in responses_data:
            question_text = response.get('question', '')
            matching_question = next((q for q in questions_data if q.get('question') == question_text), {})
            
            prompt_data['qa_pairs'].append({
                'question': question_text,
                'context': matching_question.get('context', ''),
                'response': response.get('response', ''),
                'key_points': response.get('key_points', [])
            })
        
        return prompt_data
    except Exception as e:
        logger.error(f"Failed to read client type data from {client_dir}: {e}", exc_info=True)
        return None

def analyze_client_type(client: genai.Client, prompt_data: Dict[str, Any]) -> ClientTypeAnalysis:
    logger.info(f"Analyzing client type: {prompt_data['client_type']}")
    
    try:
        qa_sample = prompt_data['qa_pairs']
        if len(qa_sample) > 10:
            step = len(qa_sample) // 10
            qa_sample = qa_sample[::step][:10]
        
        qa_sample_formatted = ""
        for i, qa in enumerate(qa_sample, 1):
            qa_sample_formatted += f"""
Example {i}:
Question: {qa['question']}
Context: {qa['context']}
Response: {qa['response']}
Key Points: {', '.join(qa['key_points'])}
"""
        
        prompt = f"""
        You are an expert financial services analyst tasked with evaluating the quality of 
        AI-generated client personas, questions, and responses for a BFSI (Banking, Financial 
        Services, and Insurance) sales chatbot.
        
        Analyze the following client type, description, and question-answer samples. Provide 
        an in-depth assessment of their quality, relevance, and effectiveness for a sales context.
        
        Client Type: {prompt_data['client_type']}
        
        Description:
        {prompt_data['description']}
        
        Sample Questions and Responses:
        {qa_sample_formatted}
        
        Your analysis should include:
        1. Evaluation of the client type description (quality rating 1-10, detailed feedback)
        2. Evaluation of the questions (quality rating 1-10, detailed feedback)
        3. Evaluation of the responses (quality rating 1-10, detailed feedback)
        4. Specific suggestions for improvement
        
        Focus on assessing:
        - Relevance to financial services sales
        - Accuracy and depth of financial knowledge
        - Appropriateness for the client type
        - Natural language quality
        - Sales effectiveness
        - Potential issues or concerns
        
        Format your response as a valid JSON object with the following structure:
        {{
            "description_quality": <rating 1-10>,
            "description_feedback": "<detailed feedback>",
            "question_quality": <rating 1-10>,
            "question_feedback": "<detailed feedback>",
            "response_quality": <rating 1-10>,
            "response_feedback": "<detailed feedback>",
            "improvement_suggestions": ["suggestion1", "suggestion2", ...]
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=8192
            )
        )
        
        result = response.text.strip()
        
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            logger.error(f"Failed to extract JSON from response: {result}")
            return ClientTypeAnalysis(
                client_type=prompt_data['client_type'],
                description_quality=5,
                description_feedback="Analysis failed to return structured data.",
                question_quality=5,
                question_feedback="Analysis failed to return structured data.",
                response_quality=5,
                response_feedback="Analysis failed to return structured data.",
                improvement_suggestions=["Retry analysis with a different model or settings."]
            )
        
        json_str = result[start_idx:end_idx]
        analysis_data = json.loads(json_str)
        
        return ClientTypeAnalysis(
            client_type=prompt_data['client_type'],
            description_quality=analysis_data['description_quality'],
            description_feedback=analysis_data['description_feedback'],
            question_quality=analysis_data['question_quality'],
            question_feedback=analysis_data['question_feedback'],
            response_quality=analysis_data['response_quality'],
            response_feedback=analysis_data['response_feedback'],
            improvement_suggestions=analysis_data['improvement_suggestions']
        )
    except Exception as e:
        logger.error(f"Failed to analyze client type {prompt_data['client_type']}: {e}", exc_info=True)
        return ClientTypeAnalysis(
            client_type=prompt_data['client_type'],
            description_quality=5,
            description_feedback=f"Analysis error: {str(e)}",
            question_quality=5,
            question_feedback="Could not analyze questions due to an error.",
            response_quality=5,
            response_feedback="Could not analyze responses due to an error.",
            improvement_suggestions=["Review the logs and retry the analysis."]
        )

def create_overall_analysis(
    client: genai.Client, 
    username: str, 
    client_analyses: List[ClientTypeAnalysis]
) -> PromptSetAnalysis:
    logger.info(f"Creating overall analysis for user: {username}")
    
    try:
        analyses_formatted = ""
        for analysis in client_analyses:
            analyses_formatted += f"""
Client Type: {analysis.client_type}
Description Quality: {analysis.description_quality}/10
Description Feedback: {analysis.description_feedback}
Question Quality: {analysis.question_quality}/10
Question Feedback: {analysis.question_feedback}
Response Quality: {analysis.response_quality}/10
Response Feedback: {analysis.response_feedback}
Improvement Suggestions: {', '.join(analysis.improvement_suggestions)}
"""
        
        prompt = f"""
        You are an expert financial services analyst tasked with providing an overall assessment
        of a set of AI-generated BFSI (Banking, Financial Services, and Insurance) sales chatbot prompts.
        
        Analyze the following individual client type analyses and create a comprehensive overall 
        assessment of the entire prompt set.
        
        Username: {username}
        
        Individual Client Type Analyses:
        {analyses_formatted}
        
        Your overall analysis should include:
        1. An overall quality rating (1-10)
        2. Major strengths across the entire prompt set
        3. Major weaknesses or areas for improvement
        4. Strategic improvement suggestions
        5. A brief executive summary (1-2 paragraphs)
        
        Format your response as a valid JSON object with the following structure:
        {{
            "overall_quality": <rating 1-10>,
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "improvement_suggestions": ["suggestion1", "suggestion2", ...],
            "summary": "<executive summary>"
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=16000
            )
        )
        
        result = response.text.strip()
        
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            logger.error(f"Failed to extract JSON from response: {result}")
            # Create a fallback analysis
            return PromptSetAnalysis(
                username=username,
                overall_quality=5,
                client_type_analyses=client_analyses,
                strengths=["Analysis failed to return structured data."],
                weaknesses=["Analysis failed to return structured data."],
                improvement_suggestions=["Retry analysis with a different model or settings."],
                summary="The analysis process encountered an error and could not generate a proper summary."
            )
        
        json_str = result[start_idx:end_idx]
        overall_data = json.loads(json_str)
        
        return PromptSetAnalysis(
            username=username,
            overall_quality=overall_data['overall_quality'],
            client_type_analyses=client_analyses,
            strengths=overall_data['strengths'],
            weaknesses=overall_data['weaknesses'],
            improvement_suggestions=overall_data['improvement_suggestions'],
            summary=overall_data['summary']
        )
    except Exception as e:
        logger.error(f"Failed to create overall analysis: {e}", exc_info=True)
        # Return a basic analysis in case of failure
        return PromptSetAnalysis(
            username=username,
            overall_quality=5,
            client_type_analyses=client_analyses,
            strengths=["Could not identify strengths due to an analysis error."],
            weaknesses=["Analysis process failed - review logs for details."],
            improvement_suggestions=["Review error logs and retry the analysis."],
            summary=f"The analysis process encountered an error: {str(e)}"
        )

def save_analysis_report(analysis: PromptSetAnalysis, output_file: Path):
    try:
        os.makedirs(output_file.parent, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"# BFSI Sales Bot Prompt Analysis for {analysis.username}\n\n")
            
            file.write("## Executive Summary\n\n")
            file.write(f"{analysis.summary}\n\n")
            file.write(f"**Overall Quality Rating:** {analysis.overall_quality}/10\n\n")
            
            file.write("## Overall Strengths\n\n")
            for strength in analysis.strengths:
                file.write(f"- {strength}\n")
            file.write("\n")
            
            file.write("## Overall Weaknesses\n\n")
            for weakness in analysis.weaknesses:
                file.write(f"- {weakness}\n")
            file.write("\n")
            
            file.write("## Strategic Improvement Suggestions\n\n")
            for suggestion in analysis.improvement_suggestions:
                file.write(f"- {suggestion}\n")
            file.write("\n")
            
            file.write("## Detailed Client Type Analyses\n\n")
            for client_analysis in analysis.client_type_analyses:
                file.write(f"### Client Type: {client_analysis.client_type}\n\n")
                
                file.write("#### Description Analysis\n")
                file.write(f"**Quality Rating:** {client_analysis.description_quality}/10\n\n")
                file.write(f"{client_analysis.description_feedback}\n\n")
                
                file.write("#### Question Analysis\n")
                file.write(f"**Quality Rating:** {client_analysis.question_quality}/10\n\n")
                file.write(f"{client_analysis.question_feedback}\n\n")
                
                file.write("#### Response Analysis\n")
                file.write(f"**Quality Rating:** {client_analysis.response_quality}/10\n\n")
                file.write(f"{client_analysis.response_feedback}\n\n")
                
                file.write("#### Improvement Suggestions\n")
                for suggestion in client_analysis.improvement_suggestions:
                    file.write(f"- {suggestion}\n")
                file.write("\n" + "-" * 80 + "\n\n")
        
        logger.info(f"Successfully saved analysis report to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save analysis report: {e}", exc_info=True)
        raise

def analyze_prompts(username: str, prompts_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
    try:
        logger.info(f"Starting analysis for user: {username}")
        
        base_dir = Path(__file__).resolve().parent.parent
        if prompts_dir is None:
            prompts_dir = base_dir / "prompts"
        if output_dir is None:
            output_dir = base_dir / "analysis"
        
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        if Path(prompts_dir).is_dir():
            client_type_dirs = [d for d in Path(prompts_dir).iterdir() if d.is_dir()]
            
            if not client_type_dirs:
                user_dir_glob = list(Path(prompts_dir).glob(f"{username}*"))
                if user_dir_glob:
                    for user_dir in user_dir_glob:
                        if user_dir.is_dir():
                            client_type_dirs.extend([d for d in user_dir.iterdir() if d.is_dir()])
        
        if not client_type_dirs:
            logger.error(f"No client type directories found in {prompts_dir}")
            raise FileNotFoundError(f"No client type directories found in {prompts_dir}")
        
        logger.info(f"Found {len(client_type_dirs)} client type directories to analyze")
        
        client = init_llm()
        
        client_analyses = []
        for client_dir in client_type_dirs:
            prompt_data = read_client_type_data(client_dir)
            
            if not prompt_data:
                continue
                
            analysis = analyze_client_type(client, prompt_data)
            client_analyses.append(analysis)
        
        if not client_analyses:
            logger.error("No client types could be analyzed")
            raise ValueError("No client types could be analyzed")
        
        overall_analysis = create_overall_analysis(client, username, client_analyses)
        
        timestamp = int(datetime.now().timestamp())
        report_file = output_dir / f"{username}_analysis_{timestamp}.md"
        save_analysis_report(overall_analysis, report_file)
        
        logger.info(f"Analysis completed for user: {username}. Report saved to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Failed to analyze prompts for user {username}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prompts for a BFSI sales bot.")
    parser.add_argument("--username", type=str, required=True, help="Username to analyze prompts for.")
    parser.add_argument("--prompts_dir", type=str, help="Directory containing prompt files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save analysis reports.")
    
    args = parser.parse_args()
    
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        report_file = analyze_prompts(args.username, prompts_dir, output_dir)
        print(f"Analysis completed successfully. Report saved to: {report_file}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"Analysis failed: {e}")
        exit(1)