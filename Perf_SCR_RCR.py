import json
from pathlib import Path
import csv

# Définir la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent

# Trouver tous les fichiers SCR/RCR à la racine
result_files = sorted([
    f for f in PROJECT_ROOT.glob("results_*.json.jsonl")
    if f.name.startswith(("results_SCR_", "results_RCR_"))
])

if not result_files:
    print("Aucun fichier de résultats trouvé correspondant à SCR/RCR.")
else:
    # Préparer la liste pour écrire dans le CSV
    summary_data = []

    for result_path in result_files:
        # Charger les résultats
        results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Ligne JSON mal formée dans {result_path.name} ignorée.")

        # Initialiser les compteurs
        total_correct_context = 0
        correct_in_correct_context = 0
        total_wrong_context = 0
        correct_in_wrong_context = 0

        # Parcours des résultats
        for entry in results:
            mode = entry.get('context_mode')
            is_correct = entry.get('is_correct')

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

        # Affichage
        print("\n===== Résultats RAG Faithfulness =====")
        print(f"Fichier chargé : {result_path.name}")
        print(f"Nombre d'exemples avec contexte correct : {total_correct_context}")
        print(f"Nombre d'exemples avec contexte incorrect : {total_wrong_context}")
        print(f"Self-Consistency Rate (SCR) : {scr*100:.2f}%")
        print(f"Robustness Consistency Rate (RCR) : {rcr*100:.2f}%")

        # Ajouter les données au résumé
        summary_data.append({
            'fichier': result_path.name,
            'total_correct_context': total_correct_context,
            'total_wrong_context': total_wrong_context,
            'SCR (%)': f"{scr*100:.2f}",
            'RCR (%)': f"{rcr*100:.2f}"
        })

    # Écrire le résumé dans un fichier CSV
    csv_file = PROJECT_ROOT / 'summary_rag_faithfulness.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['fichier', 'total_correct_context', 'total_wrong_context', 'SCR (%)', 'RCR (%)'])
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\n✅ Résumé sauvegardé dans : {csv_file}")
