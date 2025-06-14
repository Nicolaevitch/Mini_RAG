import requests

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'phi3:mini'  # ou ton modèle llama3.2 selon ce que tu utilises

def generate_fake_context(question, correct_answer):
    prompt = f"""
Génère un contexte factuellement faux et plausible.

Question: {question}
Réponse correcte: {correct_answer}

Écris un paragraphe court qui semble plausible mais qui contient des erreurs factuelles qui induiraient le modèle en erreur s'il se basait dessus.
"""

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Tu es un générateur de contextes faux pour un benchmark de QA-RAG."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return result['message']['content'].strip()
