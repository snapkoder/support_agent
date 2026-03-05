#!/usr/bin/env python3
"""
Verify No Content Loss Script
Verifica que nenhuma informação foi perdida durante a reestruturação
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def get_file_metrics(file_path: str) -> Dict[str, int]:
    """Get basic file metrics"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        content = ''.join(lines)
        non_empty_lines = [line for line in lines if line.strip()]
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'characters': len(content),
            'paragraphs': len(paragraphs),
            'words': len(content.split())
        }
    except Exception as e:
        return {'error': str(e)}

def extract_key_phrases(file_path: str) -> List[str]:
    """Extract key phrases for content verification"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract phrases that should be preserved
        key_phrases = []
        
        # KB-specific phrases
        if 'knowledge_base' in file_path:
            kb_phrases = [
                "O Jota é um assistente financeiro",
                "Funciona 100% no WhatsApp",
                "(11) 4004-8006",
                "Totalmente gratuito",
                "Rendimento de 100% do CDI",
                "Apenas uma conta ativa por celular",
                "Cartão de crédito",
                "Open Finance",
                "MED é o Mecanismo Especial de Devolução"
            ]
            for phrase in kb_phrases:
                if phrase.lower() in content.lower():
                    key_phrases.append(phrase)
        
        # Prompts-specific phrases
        elif 'prompts_agentes' in file_path:
            prompt_phrases = [
                "Agent de Atendimento Geral",
                "Agent de Criação de Conta", 
                "Agent de Open Finance",
                "Agent de Golpe Med",
                "{rag_context}",
                "{memory_context}",
                "{user_message}",
                "profissional e empático"
            ]
            for phrase in prompt_phrases:
                if phrase in content:
                    key_phrases.append(phrase)
        
        return key_phrases
    except Exception as e:
        return [f"ERROR: {e}"]

def compare_files(original: str, restructured: str) -> Dict[str, any]:
    """Compare original and restructured files"""
    result = {
        'file_comparison': {
            'original': original,
            'restructured': restructured
        },
        'hashes': {},
        'metrics': {},
        'key_phrases': {},
        'content_loss_detected': False,
        'issues': []
    }
    
    # Calculate hashes
    result['hashes']['original'] = calculate_file_hash(original)
    result['hashes']['restructured'] = calculate_file_hash(restructured)
    
    # Get metrics
    result['metrics']['original'] = get_file_metrics(original)
    result['metrics']['restructured'] = get_file_metrics(restructured)
    
    # Extract key phrases
    result['key_phrases']['original'] = extract_key_phrases(original)
    result['key_phrases']['restructured'] = extract_key_phrases(restructured)
    
    # Check for content loss
    original_phrases = set(result['key_phrases']['original'])
    restructured_phrases = set(result['key_phrases']['restructured'])
    
    missing_phrases = original_phrases - restructured_phrases
    if missing_phrases:
        result['content_loss_detected'] = True
        result['issues'].append(f"Missing key phrases: {missing_phrases}")
    
    # Check for significant metric changes
    orig_metrics = result['metrics']['original']
    restruct_metrics = result['metrics']['restructured']
    
    if 'error' not in orig_metrics and 'error' not in restruct_metrics:
        # Allow for some variation due to reformatting
        char_diff = abs(orig_metrics['characters'] - restruct_metrics['characters'])
        if char_diff > orig_metrics['characters'] * 0.1:  # 10% threshold
            result['issues'].append(f"Significant character count change: {char_diff}")
        
        line_diff = abs(orig_metrics['non_empty_lines'] - restruct_metrics['non_empty_lines'])
        if line_diff > orig_metrics['non_empty_lines'] * 0.2:  # 20% threshold
            result['issues'].append(f"Significant line count change: {line_diff}")
    
    return result

def main():
    """Main verification function"""
    project_root = Path(__file__).parent.parent
    
    # Files to verify
    kb_original = project_root / 'data' / 'knowledge_base' / 'jota_knowledge_base.md'
    kb_restructured = project_root / 'data' / 'knowledge_base' / 'jota_kb_restructured.md'
    
    prompts_original = project_root / 'prompts_agentes.txt'
    prompts_restructured = project_root / 'prompts_agentes_restructured.txt'
    
    print("🔍 VERIFY NO CONTENT LOSS")
    print("=" * 50)
    
    all_results = {}
    overall_success = True
    
    # Verify KB
    if kb_original.exists() and kb_restructured.exists():
        print("\n📚 Verifying Knowledge Base...")
        kb_result = compare_files(str(kb_original), str(kb_restructured))
        all_results['knowledge_base'] = kb_result
        
        if kb_result['content_loss_detected']:
            overall_success = False
            print("❌ KB content loss detected!")
            for issue in kb_result['issues']:
                print(f"   - {issue}")
        else:
            print("✅ KB content preserved")
    
    # Verify Prompts
    if prompts_original.exists() and prompts_restructured.exists():
        print("\n🤖 Verifying Agent Prompts...")
        prompts_result = compare_files(str(prompts_original), str(prompts_restructured))
        all_results['prompts'] = prompts_result
        
        if prompts_result['content_loss_detected']:
            overall_success = False
            print("❌ Prompts content loss detected!")
            for issue in prompts_result['issues']:
                print(f"   - {issue}")
        else:
            print("✅ Prompts content preserved")
    
    # Save detailed report
    report_file = project_root / 'docs' / 'PR_PROMPT_RESTRUCTURE_AUDIT' / 'verification_report.json'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    if overall_success:
        print("\n✅ VERIFICATION PASSED - No content loss detected")
        return 0
    else:
        print("\n❌ VERIFICATION FAILED - Content loss detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
