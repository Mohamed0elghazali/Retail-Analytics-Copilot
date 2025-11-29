import os
import sys
import json
import argparse

from typing import List, Dict, Any, Iterator

from agent.graph_hybrid import invoke_agent


def main():
    parser = argparse.ArgumentParser(
        description='Run hybrid agent with various input modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl'''
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--batch', type=str, help='Process input in batch mode with input file name')
    parser.add_argument('--out', type=str, required=True, help='Output file path')

    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.batch:
        input_file = args.batch
        if not os.path.exists(input_file):
            parser.error(f"Input file '{input_file}' not found for batch mode")
        print(f"Running in BATCH mode")
        print(f"Input file: {input_file}")
    
    print(f"Output path: {args.out}")
    process_agent(args)

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of JSON objects from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If there's invalid JSON in any line
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:  
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_number}: {e}",
                    e.doc, e.pos
                )
    return data

def save_jsonl_file(data: List[Dict[str, Any]], file_path: str, mode: str = 'w') -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        data (List[Dict[str, Any]]): List of dictionaries to save
        file_path (str): Path where to save the JSONL file
        mode (str): File mode - 'w' for write, 'a' for append
        
    Raises:
        ValueError: If data is not a list of dictionaries
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(file_path, mode, encoding='utf-8') as file:
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"All items must be dictionaries. Found: {type(item)}")
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_agent(args):
    """
    Process the agent based on the provided arguments
    """
    print(f"Processing batch from {args.batch}")
    try:
        data = read_jsonl_file(args.batch)
        results = []
        for record in data:
            try:
                result = invoke_agent(record.get("id"), record.get("question"), record.get("format_hint"))
            except Exception as e:
                print(f"Error during processing: {e}")
                result = {"id": record.get("id"), "final_answer": str(e)}
            
            results.append(result)

    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        save_jsonl_file(results, args.out)
        print("Processing completed successfully!")
        sys.exit(1)

if __name__ == '__main__':
    main()

