Readme

Telechargement dataset QA : 
Script load_dataset.py

Telechargement LLM :
lancer ./dowload_llm.sh

Creation rag : script « create_rag”

-	Simule les 2 cas : injection de contextes corrects ou incorrects.
-	Compare la réponse du modèle à la réponse attendue.
-	Produit un fichier rag_faithfulness_results.json contenant toutes les prédictions et leurs évaluations.


Performance SCR/RCR : script « perf_scr_rcr”
-	Lire ton fichier de résultats.
-	Faire la séparation context_mode = correct / incorrect.
-	Calculer et afficher SCR et RCR.


Que pourrait-on ajouter pour aller vers une pipeline complète de publication scientifique (optionnel) :
Étape	Utile pour	Implémentable
🔄 Répéter plusieurs runs (stochasticity)	Avoir des barres d'erreur	✅ très simple à ajouter
📊 Grapher les performances	Générer des courbes SCR/RCR	✅ avec matplotlib
📄 Logger les hyperparamètres	Faciliter la reproductibilité	✅ très simple
