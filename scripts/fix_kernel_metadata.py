#!/usr/bin/env python3
"""
Fix Kaggle notebook - Add kernel metadata
"""
import json
from pathlib import Path

def add_kernel_metadata(notebook_path):
    """Add kernel metadata to notebook for Kaggle compatibility"""
    
    print(f"Reading notebook: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Ensure metadata exists
    if 'metadata' not in nb:
        nb['metadata'] = {}
    
    print("\nAdding kernel metadata...")
    
    # Add kernelspec for Python 3
    nb['metadata']['kernelspec'] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    print("  - kernelspec: python3")
    
    # Add language_info
    nb['metadata']['language_info'] = {
        "name": "python",
        "version": "3.10.12",
        "mimetype": "text/x-python",
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py"
    }
    print("  - language_info: python 3.10.12")
    
    # Add Kaggle-specific metadata
    nb['metadata']['kaggle'] = {
        "accelerator": "gpu",
        "dataSources": [],
        "dockerImageVersionId": 30699,
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
        "isGpuEnabled": True
    }
    print("  - kaggle metadata: GPU enabled, internet enabled")
    
    # Backup original
    backup_path = Path(str(notebook_path) + '.backup2')
    print(f"\nCreating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    # Save updated notebook
    print(f"Saving updated notebook: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("Kernel metadata added successfully!")
    print("="*70)
    print(f"Kernelspec: {nb['metadata']['kernelspec']['name']}")
    print(f"Language: {nb['metadata']['language_info']['name']} {nb['metadata']['language_info']['version']}")
    print(f"GPU enabled: {nb['metadata']['kaggle']['isGpuEnabled']}")
    print(f"Internet enabled: {nb['metadata']['kaggle']['isInternetEnabled']}")
    print(f"Backup saved: {backup_path}")
    print("="*70)

if __name__ == "__main__":
    notebook_path = Path("scripts/finetune_qwen_lora_kaggle.v1.0.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        exit(1)
    
    add_kernel_metadata(notebook_path)
    print("\nNotebook is now ready for Kaggle!")
