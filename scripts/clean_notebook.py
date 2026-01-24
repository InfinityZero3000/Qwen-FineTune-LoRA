#!/usr/bin/env python3
"""
Clean Kaggle notebook - Remove emojis and unicode symbols
"""
import json
import re
import sys
from pathlib import Path

def clean_text(text):
    """Remove emojis and special unicode symbols"""
    # Remove all emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove specific unicode symbols
    replacements = {
        '🚨': '',
        '⚠️': '',
        '✓': '',
        '✗': '',
        '⚡': '',
        '🎯': '',
        '🔧': '',
        '✅': '',
        '📁': '',
        '📦': '',
        '📋': '',
        '💡': '',
        '🔥': '',
        '⏰': '',
        '📊': '',
        '📈': '',
        '🎉': '',
        '💪': '',
        '🏃‍♂️': '',
        '💻': '',
        '🖥️': '',
        '📱': '',
        '🌟': '',
        '🎓': '',
        '📚': '',
        '📝': '',
        '✨': '',
        '🔍': '',
        '└─': '',
        '→': '->',
        '×': 'x',
        '↓': 'down',
        '•': '-',
        '⚙️': '',
        '1️⃣': '1.',
        '2️⃣': '2.',
        '3️⃣': '3.',
        '4️⃣': '4.',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def main():
    notebook_path = Path("scripts/finetune_qwen_lora_kaggle.v1.0.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        sys.exit(1)
    
    print(f"Reading notebook: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clean all cells
    total_cells = len(notebook['cells'])
    cleaned_cells = 0
    
    for i, cell in enumerate(notebook['cells']):
        if 'source' in cell:
            original = ''.join(cell['source'])
            cleaned = [clean_text(line) for line in cell['source']]
            cleaned_text = ''.join(cleaned)
            
            if original != cleaned_text:
                cell['source'] = cleaned
                cleaned_cells += 1
                print(f"  Cell {i+1}: Cleaned")
    
    print(f"\nProcessed {total_cells} cells, cleaned {cleaned_cells} cells")
    
    # Backup original
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    # Save cleaned notebook
    print(f"Saving cleaned notebook: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("\nNotebook cleaned successfully!")
    print("- All emojis removed")
    print("- Unicode symbols cleaned")
    print("- Special characters normalized")
    print(f"- Backup saved: {backup_path}")

if __name__ == "__main__":
    main()
