**Débutant**

- Qu'est-ce qu'un réservoir ?

- Qu'est-ce que le reservoir computing ?

- Qu'est-ce que le readout ?

- Pourquoi le nom de "réservoir" ?

- Pourquoi le nom de "readout" ?

- Sur quelles tâches le reservoir computing est bon ? 

- Sur quelles tâches le reservoir computing est mauvais ?

- Combien de neurones utiliser environ ? (10, 100, 1000, 1 million ?)

- À quoi sert le ridge dans le readout ?

- Comment sont fixés les poids dans le réservoir ?

- Est-ce qu'il y a un apprentissage des poids de l'entrée vers les neurones du réservoir ?

- Crée un dataset sur la série temporelle Mackey-Glass normalisée, avec une prédiction à 20 pas de temps (import de Mackey-Glass, normalisation, séparation X/Y, train/test, etc)

- Crée un réservoir/ESN simple, et entraîne-le sur un jeu de données qui contient plusieurs séries temporelles (avec le noeud ESN ou Reservoir+Ridge)

- Crée un echo state network avec parallélisation

**Intermédiaire**

- Quelle est la différence entre "echo state network" et "reservoir computing" ?

- Est-ce qu'il existe d'autres formes de reservoir computing ?

- Pourquoi on parle de "computing at the edge of chaos" ?

- Qu'est-ce que c'est l' "echo state property" ?

- Quel est le papier qui introduit le reservoir computing / echo state network ?

- Quel est le papier qui introduit l'echo state network ?

- Quels sont tous les hyper-paramètres ?

- Comment choisir les hyper-paramètres ?

- Écris un code pour afficher l'évolution des neurones du réservoir sur la série Lorenz

- Crée un modèle de NVAR avec un apprentissage online

- Crée un réservoir dans lequel tous les neurones sont connectés en ligne, et l'entrée est connectée au premier neurone

- Crée un modèle DeepESN

- Crée un modèle avec 10 réservoirs en parallèle connectés au même readout

**Avancé**


- Qu'est-ce qu'un liquid state machine ?

- Quelle est l'explicabilité des modèles de reservoir computing ?

- À quel point les résultats varient entre deux réservoirs initialisés différemment ?

- Quelle est l'influence de la sparsité de la matrice des poids sur les performances ?

- Crée un node ReservoirPy qui rajoute un bruit gaussien à l'entrée qu'il reçoit

- Écris une recherche d'hyper-paramètres avec recherche avec du TPE sampler, sur 300 instances, et en évaluant la NRMSE, le R² et l'erreur maximale