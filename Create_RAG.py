import os
import json
import requests
import random
import re

# Configuration générale
OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'phi3:mini'  # ou 'mistral' selon ton modèle installé
DATASET_DIR = './mini_benchmark_data'
EVAL_RESULTS = './rag_faithfulness_results.json'

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

# Fonction principale d'évaluation
def evaluate_dataset(filename):
    filepath = os.path.join(DATASET_DIR, filename)
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        item = json.loads(line)

        question = item['question']
        gold_answer = str(item.get('answer', item.get('correct_answer')))

        # On choisit aléatoirement d'injecter un contexte correct ou incorrect
        mode = random.choice(['correct', 'incorrect'])

        if mode == 'correct' and 'correct_doc' in item:
            context = item['correct_doc']
        elif mode == 'incorrect' and 'wrong_doc' in item:
            context = item['wrong_doc']
        elif 'fake_context' in item:
            context = item['fake_context']
        else:
            context = ''  # fallback: aucun contexte

        try:
            model_answer = query_ollama(question, context)
            is_correct = normalize_answer(model_answer) == normalize_answer(gold_answer)
            print(f"[{idx+1}] Mode: {mode} | Réponse correcte: {is_correct}")
        except Exception as e:
            print(f"Erreur à l'exemple {idx+1} : {e}")
            model_answer = "ERROR"
            is_correct = False

        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'context_mode': mode,
            'context_used': context[:200],  # juste un aperçu du contexte
            'model_answer': model_answer,
            'is_correct': is_correct
        })

    return results

# Liste des datasets qu'on va évaluer (tu peux en ajouter d'autres ici)
datasets = [
    'clasheval_sample.json',
    'redditqa_sample.json',
    'triviaqa_sample_augmented.json',
    'naturalqa_sample_augmented.json'
]

# On lance l'évaluation
final_results = []

for dataset in datasets:
    print(f"\n--- Évaluation sur le dataset: {dataset} ---")
    res = evaluate_dataset(dataset)
    final_results.extend(res)

# Sauvegarde des résultats
with open(EVAL_RESULTS, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Évaluation terminée ! Résultats sauvegardés dans {EVAL_RESULTS}")
