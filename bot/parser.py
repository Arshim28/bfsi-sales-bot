import os
import time
import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud_services import LlamaParse

from .utils import setup_logger

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = setup_logger(__name__)

async def parse_document_async(input_file_path: str, output_file_path: str) -> str:
    """Async version of document parsing using aload_data"""
    logger.info(f"Starting to parse document: {input_file_path}")
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
    )

    try:
        # Use the async version of load_data
        # aload_data returns a list of Document objects
        documents = await parser.aload_data(input_file_path)
        
        # Concatenate content from all documents
        all_content = []
        for doc in documents:
            all_content.append(doc.text) # Assuming .text attribute holds the content
        content = "\n\n".join(all_content)

        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(content)
        logger.info(f"Successfully parsed and saved document to: {output_file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to parse or save document {input_file_path}", exc_info=True)
        raise

def parse_document(input_file_path: str, output_file_path: str) -> str:
    """Synchronous wrapper for backward compatibility"""
    logger.info(f"Starting to parse document: {input_file_path}")
    try:
        # Use nest_asyncio to run the async function in sync context
        import nest_asyncio
        nest_asyncio.apply()
        
        # Run the async function in the current event loop
        return asyncio.get_event_loop().run_until_complete(
            parse_document_async(input_file_path, output_file_path)
        )
    except Exception as e:
        logger.error(f"Failed to parse or save document {input_file_path}", exc_info=True)
        raise

async def main_async(args):
    """Async version of the main function"""
    output_dir = Path(__file__).resolve().parent.parent / "data"
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}", exc_info=True)
        raise

    kb_output_filename = f"{args.username}_knowledge_base.md"
    persona_output_filename = f"{args.username}_agent_persona.md"

    kb_output_path = output_dir / kb_output_filename
    persona_output_path = output_dir / persona_output_filename

    logger.info(f"Attempting to parse knowledge base document: {args.knowledge_base_file}...")
    try:
        await parse_document_async(str(args.knowledge_base_file), str(kb_output_path))
        logger.info(f"Successfully processed knowledge base. Output saved to: {kb_output_path}")
    except Exception as e: 
        logger.error(f"Error processing knowledge base document '{args.knowledge_base_file}'", exc_info=True)

    logger.info(f"Attempting to parse agent persona document: {args.agent_persona_file}...")
    try:
        await parse_document_async(str(args.agent_persona_file), str(persona_output_path))
        logger.info(f"Successfully processed agent persona. Output saved to: {persona_output_path}")
    except Exception as e: 
        logger.error(f"Error processing agent persona document '{args.agent_persona_file}'", exc_info=True)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Parse knowledge base and agent persona documents.")
    arg_parser.add_argument("--username", type=str, required=True, help="Username for naming the output files.")
    arg_parser.add_argument("--knowledge_base_file", type=str, required=True, help="Path to the knowledge base document.")
    arg_parser.add_argument("--agent_persona_file", type=str, required=True, help="Path to the agent persona document.")

    args = arg_parser.parse_args()
    
    # Apply nest_asyncio for the command-line use case
    import nest_asyncio
    nest_asyncio.apply()
    
    # Run the async main function
    asyncio.run(main_async(args))
