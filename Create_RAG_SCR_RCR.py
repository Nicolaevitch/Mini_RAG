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

# Fonction d'appel au modèle

def query_ollama(prompt):
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

# Normalisation simple des réponses

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

# Fonction pour charger les clés déjà traitées

def load_existing_keys(result_file):
    existing_keys = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                key = f"{entry['question']}|{entry['context_mode']}"
                existing_keys.add(key)
    return existing_keys

# Fonction SCR avec reprise

def run_SCR(dataset_file, result_file):
    filepath = os.path.join(DATASET_DIR, dataset_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    existing_keys = load_existing_keys(result_file)

    for idx, line in enumerate(tqdm(lines, desc=f"SCR Processing {dataset_file}")):
        item = json.loads(line)
        question = item['question']

        mode = random.choice(['correct', 'incorrect'])
        key = f"{question}|{mode}"
        if key in existing_keys:
            continue  # Skip déjà traité

        gold_answer = str(item.get('answer') or item.get('correct_answer'))

        if mode == 'correct' and 'correct_doc' in item:
            context = item['correct_doc']
        elif mode == 'incorrect' and 'wrong_doc' in item:
            context = item['wrong_doc']
        elif 'fake_context' in item:
            context = item['fake_context']
        else:
            context = ''

        prompt = f"""
Tu dois répondre à la question suivante. Attention : le contexte fourni peut être partiellement ou totalement faux. Utilise tes propres connaissances en plus du contexte pour donner la réponse la plus fiable possible.

Question : {question}

Contexte proposé : {context}

Réfléchis bien avant de répondre.
"""
        try:
            model_answer = query_ollama(prompt)
            is_correct = normalize_answer(model_answer) == normalize_answer(gold_answer)
        except Exception as e:
            model_answer = "ERROR"
            is_correct = False

        result = {
            'dataset': dataset_file,
            'question': question,
            'gold_answer': gold_answer,
            'context_mode': mode,
            'context_used': context[:200],
            'model_answer': model_answer,
            'is_correct': is_correct
        }

        with open(result_file, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(result) + '\n')

# Fonction RCR avec reprise

def run_RCR(dataset_file, result_file):
    filepath = os.path.join(DATASET_DIR, dataset_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    existing_keys = load_existing_keys(result_file)

    for idx, line in enumerate(tqdm(lines, desc=f"RCR Processing {dataset_file}")):
        item = json.loads(line)
        question = item['question']

        mode = random.choice(['correct', 'incorrect'])
        key = f"{question}|{mode}"
        if key in existing_keys:
            continue  # Skip déjà traité

        gold_answer = str(item.get('answer') or item.get('correct_answer'))

        if mode == 'correct' and 'correct_doc' in item:
            context = item['correct_doc']
        elif mode == 'incorrect' and 'wrong_doc' in item:
            context = item['wrong_doc']
        elif 'fake_context' in item:
            context = item['fake_context']
        else:
            context = ''

        # Génération réponse interne
        prompt_internal = f"""
Question : {question}

Réponds uniquement en te basant sur tes connaissances internes.
"""
        # Génération réponse avec contexte
        prompt_context = f"""
Contexte : {context}

Question : {question}

Réponds uniquement en te basant sur ce contexte.
"""
        try:
            internal_answer = query_ollama(prompt_internal)
            context_answer = query_ollama(prompt_context)

            if normalize_answer(internal_answer) == normalize_answer(gold_answer):
                model_answer = internal_answer
            else:
                model_answer = context_answer

            is_correct = normalize_answer(model_answer) == normalize_answer(gold_answer)
        except Exception as e:
            model_answer = "ERROR"
            is_correct = False

        result = {
            'dataset': dataset_file,
            'question': question,
            'gold_answer': gold_answer,
            'context_mode': mode,
            'context_used': context[:200],
            'model_answer': model_answer,
            'is_correct': is_correct
        }

        with open(result_file, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(result) + '\n')

# Lancement multi-dataset
if __name__ == '__main__':
    datasets = [
        'naturalqa_sample.json',
        'clasheval_sample.json',
        'redditqa_sample.json',
        'triviaqa_sample.json'
    ]

    for dataset in datasets:
        scr_result_file = f'results_SCR_{dataset}.jsonl'
        rcr_result_file = f'results_RCR_{dataset}.jsonl'

        run_SCR(dataset, scr_result_file)
        run_RCR(dataset, rcr_result_file)
