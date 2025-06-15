import os
import json
import requests
import random
import re
from tqdm import tqdm

# Configuration générale
OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'phi3:mini'
DATASET_DIR = './mini_benchmark_data'
EVAL_RESULTS = './rag_faithfulness_results.jsonl'

# Normalisation simple des réponses pour comparer
def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

# Fonction d'appel à Ollama
def query_ollama(question, context):
    prompt = f"""
Contexte : {context}

Question : {question}

En te basant uniquement sur le contexte et tes connaissances internes, donne la réponse la plus factuelle possible à la question.
"""
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un assistant expert en question-answering."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return result['message']['content'].strip()

# Chargement des résultats déjà existants (si le fichier existe)
def load_existing_results():
    existing = {}
    if os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                existing_key = (item['dataset'], item['question'])
                existing[existing_key] = item
    return existing

# Fonction principale d'évaluation

def evaluate_dataset(filename, existing_results):
    filepath = os.path.join(DATASET_DIR, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    to_process = []

    for idx, line in enumerate(lines):
        item = json.loads(line)
        key = (filename, item['question'])
        if key not in existing_results:
            to_process.append((idx, item))

    print(f"{len(to_process)}/{total} exemples restants à traiter dans {filename}")

    for idx, item in tqdm(to_process, desc=f"Traitement {filename}"):
        question = item['question']
        gold_answer = str(item.get('answer') or item.get('correct_answer'))

        mode = random.choice(['correct', 'incorrect'])
        if mode == 'correct' and 'correct_doc' in item:
            context = item['correct_doc']
        elif mode == 'incorrect' and 'wrong_doc' in item:
            context = item['wrong_doc']
        elif 'fake_context' in item:
            context = item['fake_context']
        else:
            context = ''

        try:
            model_answer = query_ollama(question, context)
            is_correct = normalize_answer(model_answer) == normalize_answer(gold_answer)
            print(f"\n[Ex {idx+1}] Mode: {mode} | Correct: {is_correct}")
        except Exception as e:
            print(f"Erreur à l'exemple {idx+1} : {e}")
            model_answer = "ERROR"
            is_correct = False

        result = {
            'dataset': filename,
            'question': question,
            'gold_answer': gold_answer,
            'context_mode': mode,
            'context_used': context[:200],
            'model_answer': model_answer,
            'is_correct': is_correct
        }

        with open(EVAL_RESULTS, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

# Liste des datasets à traiter
datasets = [
    'clasheval_sample.json',
    'redditqa_sample.json',
    'triviaqa_sample_augmented.json',
    'naturalqa_sample_augmented.json'
]

existing_results = load_existing_results()
total_done = len(existing_results)
print(f"\n👉 Déjà traités au total : {total_done}")

for dataset in datasets:
    print(f"\n--- Évaluation sur le dataset: {dataset} ---")
    evaluate_dataset(dataset, existing_results)

print("\n✅ Évaluation terminée ! Tous les nouveaux résultats ont été sauvegardés.")
