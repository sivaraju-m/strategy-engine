#!/usr/bin/env python3
"""
Helper script to update import statements in Python files.
Usage: python update_imports.py <directory>
"""
import os
import re
import sys

def update_imports(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Update import statements
    updated_content = re.sub(
        r'from ai_trading_machine\.(\w+)', 
        r'from strategy_engine.\1', 
        content
    )
    updated_content = re.sub(
        r'import ai_trading_machine\.(\w+)', 
        r'import strategy_engine.\1', 
        updated_content
    )
    
    # Update shared services imports
    updated_content = re.sub(
        r'from strategy_engine\.utils\.(logger|error_handling|config_parser)', 
        r'from shared_services.utils.\1', 
        updated_content
    )
    
    # Update data pipeline imports
    updated_content = re.sub(
        r'from strategy_engine\.ingest\.', 
        r'from trading_data_pipeline.ingest.', 
        updated_content
    )
    
    if content != updated_content:
        with open(file_path, 'w') as file:
            file.write(updated_content)
        return True
    return False

def process_directory(directory):
    updated_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports(file_path):
                    updated_files += 1
                    print(f"Updated imports in {file_path}")
    
    return updated_files

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    updated = process_directory(directory)
    print(f"Updated {updated} files")
