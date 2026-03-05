#!/usr/bin/env python3
"""
Update Dataset with RAG Flags Script
Adds requires_rag field to jota_test_questions.json based on 🔍 markings in markdown
"""

import json
import re
from pathlib import Path

def load_markdown_rag_questions(file_path: str) -> set:
    """Extract question IDs that have 🔍 markings from markdown"""
    rag_questions = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for questions with 🔍: - *X. Pergunta:** "Question text" 🔍
    pattern = r'-\s*\*(\d+)\.\s*Pergunta:\*\*\s*"([^"]+)"\s*🔍'
    matches = re.findall(pattern, content)
    
    print(f"Debug: Found {len(matches)} matches with pattern: {pattern}")
    print(f"Debug: Sample matches: {matches[:3]}")
    
    current_agent = 'atendimento_geral'  # Default
    
    # Track agent sections
    agent_sections = re.split(r'### Agent de', content)
    
    for match in matches:
        question_num = match[0]
        
        # Find which agent section this question belongs to
        question_text = match[1]
        position = content.find(f'*{question_num}. Pergunta:** "{question_text}"')
        
        # Find the last agent section before this question
        before_question = content[:position]
        agent_matches = re.findall(r'### Agent de\s+([^\n]+)', before_question)
        
        if agent_matches:
            current_agent = agent_matches[-1].lower().replace(' ', '_')
        
        # Create ID with agent prefix
        agent_prefix = {
            'atendimento_geral': 'AG',
            'criacao_conta': 'CC', 
            'open_finance': 'OF',
            'golpe_med': 'GM'
        }.get(current_agent, 'AG')
        
        question_id = f"{agent_prefix}{question_num.zfill(2)}"
        rag_questions.add(question_id)
        print(f"Debug: Added RAG question: {question_id} (agent: {current_agent})")
    
    return rag_questions

def load_json_dataset(file_path: str) -> list:
    """Load JSON dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_dataset_with_rag_flags(dataset: list, rag_questions: set) -> list:
    """Update dataset with requires_rag field"""
    updated_dataset = []
    
    for item in dataset:
        updated_item = item.copy()
        
        # Check if this question ID is in RAG questions
        if item['id'] in rag_questions:
            updated_item['requires_rag'] = True
            print(f"Debug: Marked {item['id']} as RAG question")
        else:
            updated_item['requires_rag'] = False
        
        updated_dataset.append(updated_item)
    
    return updated_dataset

def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    
    # File paths
    markdown_file = project_root / 'tests' / 'jota_test_questions.md'
    json_file = project_root / 'jota_test_questions.json'
    output_file = project_root / 'jota_test_questions_with_rag.json'
    
    print("🔍 Updating dataset with RAG flags...")
    
    # Load RAG questions from markdown
    rag_questions = load_markdown_rag_questions(str(markdown_file))
    print(f"Found {len(rag_questions)} questions marked with 🔍 in markdown")
    
    # Load JSON dataset
    dataset = load_json_dataset(str(json_file))
    print(f"Loaded {len(dataset)} questions from JSON")
    
    # Update dataset
    updated_dataset = update_dataset_with_rag_flags(dataset, rag_questions)
    
    # Count RAG questions
    rag_count = sum(1 for item in updated_dataset if item.get('requires_rag', False))
    print(f"Updated dataset: {rag_count}/{len(updated_dataset)} questions marked for RAG")
    
    # Save updated dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated dataset saved to: {output_file}")
    
    # Show some examples
    print("\n📋 Sample RAG questions:")
    for item in updated_dataset[:10]:
        if item.get('requires_rag', False):
            print(f"  - {item['id']}: {item['question'][:50]}...")
    
    print(f"\n📊 Summary:")
    print(f"  - Total questions: {len(updated_dataset)}")
    print(f"  - RAG questions: {rag_count}")
    print(f"  - Non-RAG questions: {len(updated_dataset) - rag_count}")
    print(f"  - RAG usage rate: {(rag_count / len(updated_dataset)) * 100:.1f}%")

if __name__ == "__main__":
    main()
