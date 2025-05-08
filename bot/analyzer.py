import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import glob

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

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

def init_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Initialize a more advanced Google Gemini model for analysis.
    
    Args:
        temperature: Creativity parameter for the model generation.
        
    Returns:
        An instance of ChatGoogleGenerativeAI.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",  
            google_api_key=api_key,
            temperature=temperature,
            top_p=0.95,
            max_tokens=20000
        )
        logger.info("Successfully initialized Google Gemini model for analysis")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise

def read_prompt_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a prompt file.
    
    Args:
        file_path: Path to the prompt file.
        
    Returns:
        Dictionary containing client_type, description, and a list of question-response pairs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        client_type = Path(file_path).stem.replace('_prompt', '')
        
        sections = content.split('=' * 80)
        if len(sections) < 2:
            raise ValueError(f"Invalid prompt file format: {file_path}")
        
        header = sections[0].strip()
        description_lines = header.split('\n')
        if len(description_lines) < 2:
            raise ValueError(f"Invalid prompt file header: {file_path}")
        
        description = '\n'.join(description_lines[1:]).strip()
        
        qa_content = sections[1]
        qa_pairs = qa_content.split('-' * 80)
        
        pairs = []
        for pair in qa_pairs:
            pair = pair.strip()
            if not pair:
                continue
                
            question_match = re.search(r'##\s+Q\d+:\s+(.*?)\n', pair)
            context_match = re.search(r'Context:\s+(.*?)\n\n', pair)
            response_match = re.search(r'Response:\s+(.*?)\n\nKey points:', pair, re.DOTALL)
            
            if question_match and response_match:
                question = question_match.group(1).strip()
                context = context_match.group(1).strip() if context_match else ""
                response = response_match.group(1).strip()
                
                key_points_text = pair.split('Key points:')[-1].strip()
                key_points = [
                    point.lstrip('- ').strip() 
                    for point in key_points_text.split('\n') 
                    if point.strip() and point.startswith('-')
                ]
                
                pairs.append({
                    'question': question,
                    'context': context,
                    'response': response,
                    'key_points': key_points
                })
        
        return {
            'client_type': client_type,
            'description': description,
            'qa_pairs': pairs
        }
    except Exception as e:
        logger.error(f"Failed to read prompt file {file_path}: {e}", exc_info=True)
        raise

def analyze_client_type(llm: ChatGoogleGenerativeAI, prompt_data: Dict[str, Any]) -> ClientTypeAnalysis:
    """
    Analyze a client type prompt file.
    
    Args:
        llm: The language model.
        prompt_data: Dictionary containing client_type, description, and qa_pairs.
        
    Returns:
        ClientTypeAnalysis object.
    """
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert financial services analyst tasked with evaluating the quality of 
            AI-generated client personas, questions, and responses for a BFSI (Banking, Financial 
            Services, and Insurance) sales chatbot.
            
            Analyze the following client type, description, and question-answer samples. Provide 
            an in-depth assessment of their quality, relevance, and effectiveness for a sales context.
            
            Client Type: {client_type}
            
            Description:
            {description}
            
            Sample Questions and Responses:
            {qa_samples}
            
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
            """),
        ])
        
        chain = (
            prompt 
            | llm 
            | StrOutputParser()
        )
        
        result = chain.invoke({
            "client_type": prompt_data['client_type'],
            "description": prompt_data['description'],
            "qa_samples": qa_sample_formatted
        })
        
        result = result.strip()
        
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
    llm: ChatGoogleGenerativeAI, 
    username: str, 
    client_analyses: List[ClientTypeAnalysis]
) -> PromptSetAnalysis:
    """
    Create an overall analysis based on individual client type analyses.
    
    Args:
        llm: The language model.
        username: Username associated with the prompt set.
        client_analyses: List of ClientTypeAnalysis objects.
        
    Returns:
        PromptSetAnalysis object.
    """
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
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert financial services analyst tasked with providing an overall assessment
            of a set of AI-generated BFSI (Banking, Financial Services, and Insurance) sales chatbot prompts.
            
            Analyze the following individual client type analyses and create a comprehensive overall 
            assessment of the entire prompt set.
            
            Username: {username}
            
            Individual Client Type Analyses:
            {analyses}
            
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
            """),
        ])
        
        # Create the chain
        chain = (
            prompt 
            | llm 
            | StrOutputParser()
        )
        
        # Execute the chain
        result = chain.invoke({
            "username": username,
            "analyses": analyses_formatted
        })
        
        # Parse the results
        result = result.strip()
        
        # Extract JSON
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
    """
    Save the analysis report to a file.
    
    Args:
        analysis: PromptSetAnalysis object.
        output_file: Path to save the report.
    """
    try:
        # Ensure the parent directory exists
        os.makedirs(output_file.parent, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            # Write the header
            file.write(f"# BFSI Sales Bot Prompt Analysis for {analysis.username}\n\n")
            
            # Write executive summary
            file.write("## Executive Summary\n\n")
            file.write(f"{analysis.summary}\n\n")
            file.write(f"**Overall Quality Rating:** {analysis.overall_quality}/10\n\n")
            
            # Write overall strengths
            file.write("## Overall Strengths\n\n")
            for strength in analysis.strengths:
                file.write(f"- {strength}\n")
            file.write("\n")
            
            # Write overall weaknesses
            file.write("## Overall Weaknesses\n\n")
            for weakness in analysis.weaknesses:
                file.write(f"- {weakness}\n")
            file.write("\n")
            
            # Write overall improvement suggestions
            file.write("## Strategic Improvement Suggestions\n\n")
            for suggestion in analysis.improvement_suggestions:
                file.write(f"- {suggestion}\n")
            file.write("\n")
            
            # Write detailed analysis for each client type
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
    """
    Analyze all prompt files for a user.
    
    Args:
        username: Username to analyze prompts for.
        prompts_dir: Directory containing prompt files. Defaults to ../prompts/.
        output_dir: Directory to save analysis reports. Defaults to ../analysis/.
    
    Returns:
        Path to the generated analysis report.
    """
    try:
        logger.info(f"Starting analysis for user: {username}")
        
        # Set up paths
        base_dir = Path(__file__).resolve().parent.parent
        if prompts_dir is None:
            prompts_dir = base_dir / "prompts"
        if output_dir is None:
            output_dir = base_dir / "analysis"
        
        # Ensure directories exist
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all prompt files for this user
        prompt_pattern = os.path.join(prompts_dir, "*_prompt.txt")
        prompt_files = glob.glob(prompt_pattern)
        
        if not prompt_files:
            logger.error(f"No prompt files found in {prompts_dir}")
            raise FileNotFoundError(f"No prompt files found in {prompts_dir}")
        
        logger.info(f"Found {len(prompt_files)} prompt files to analyze")
        
        # Initialize LLM
        llm = init_llm()
        
        # Read and analyze each prompt file
        client_analyses = []
        for file_path in prompt_files:
            prompt_data = read_prompt_file(file_path)
            analysis = analyze_client_type(llm, prompt_data)
            client_analyses.append(analysis)
        
        # Create overall analysis
        overall_analysis = create_overall_analysis(llm, username, client_analyses)
        
        # Save analysis report
        timestamp = Path(__file__).resolve().stat().st_mtime
        report_file = output_dir / f"{username}_analysis_{int(timestamp)}.md"
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
