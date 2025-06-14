import json

# Charger les résultats générés par le premier script
RESULTS_FILE = './rag_faithfulness_results.json'

with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Initialiser les compteurs
total_correct_context = 0
correct_in_correct_context = 0

total_wrong_context = 0
correct_in_wrong_context = 0

# Parcours des résultats
for entry in results:
    mode = entry['context_mode']
    is_correct = entry['is_correct']

    if mode == 'correct':
        total_correct_context += 1
        if is_correct:
            correct_in_correct_context += 1
    elif mode == 'incorrect':
        total_wrong_context += 1
        if is_correct:
            correct_in_wrong_context += 1

# Calcul des métriques
scr = correct_in_correct_context / total_correct_context if total_correct_context > 0 else 0
rcr = correct_in_wrong_context / total_wrong_context if total_wrong_context > 0 else 0

print("\n===== Résultats RAG Faithfulness =====")
print(f"Nombre d'exemples avec contexte correct : {total_correct_context}")
print(f"Nombre d'exemples avec contexte incorrect : {total_wrong_context}")
print(f"Self-Consistency Rate (SCR) : {scr*100:.2f}%")
print(f"Robustness Consistency Rate (RCR) : {rcr*100:.2f}%")
