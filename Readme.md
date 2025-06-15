Telechargement dataset QA : 
Script load_dataset.py

Telechargement LLM :
lancer ./dowload_llm.sh

Si difficultés à lancer le fichier dowload_llm.sh écrire dans l'interface de commande : 

sudo apt update && sudo apt install dos2unix
dos2unix dowload_llm.sh
bash dowload_llm.sh

Creation rag : script « create_rag”

-	Simule les 2 cas : injection de contextes corrects ou incorrects.
-	Compare la réponse du modèle à la réponse attendue.
-	Produit un fichier rag_faithfulness_results.json contenant toutes les prédictions et leurs évaluations.


Performance SCR/RCR : script « perf_scr_rcr”
-	Lire ton fichier de résultats.
-	Faire la séparation context_mode = correct / incorrect.
-	Calculer et afficher SCR et RCR.
