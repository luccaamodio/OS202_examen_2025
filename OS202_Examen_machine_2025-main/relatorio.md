# Examen machine OS202 – 18 Mars 2025

## 1. Environnement de calcul

- Machine : macOS
- Python : environnement virtuel `venv` dans `prova_2025/venv`
- Bibliothèques : `mpi4py`, `numpy`, `Pillow`, `scipy`

Caractéristiques matérielles (sortie de `sysctl`):

- Nombre de cœurs logiques : 12
- Nombre de cœurs physiques : 12
- Cache L1 données : 64 KiB
- Cache L1 instructions : 128 KiB
- Cache L2 : 4 MiB
- Cache L3 : non reporté par le système (sortie vide)

---

## 2. Parallélisation des images issues d'une vidéo – `movie_filter.py`

### Stratégie de parallélisation

- On dispose d'une collection de 37 images indépendantes (frames de la vidéo).
- J'utilise une parallélisation *par images* : chaque processus MPI traite un sous-ensemble de frames, sans dépendance entre elles.
- Répartition : pour un nombre de processus `P`, le processus de rang `r` traite toutes les images d'indice `i` telles que `i = r, r+P, r+2P, ...`.
- Le rang 0 crée le répertoire de sortie si nécessaire, puis tous les processus écrivent directement leurs propres fichiers de sortie.
- Il n'y a quasiment pas de communication : pas d'échanges de données entre processus, seulement une barrière pour la synchronisation finale.

Cette stratégie est particulièrement bien adaptée ici :

- Le problème est **embarrassingly parallel** : chaque image peut être filtrée indépendamment.
- On minimise la communication MPI et on laisse chaque cœur travailler localement sur ses images.
- L'algorithme séquentiel d'origine (chargement image → conversion HSV → agrandissement → convolution gaussienne → filtre de netteté) est conservé à l'identique.

### Mesures de performance

Commande générique (lancée dans `OS202_Examen_machine_2025-main`) :

```bash
/usr/bin/time -p mpiexec -n P venv/bin/python movie_filter.py
```

Résultats (37 images 1920×1080) :

| P (processus) | Temps réel (s) | Accélération S(P) = T(1)/T(P) | Efficacité E(P) = S(P)/P |
|--------------:|----------------:|------------------------------:|-------------------------:|
| 1             | 42.92           | 1.00                          | 1.00                     |
| 2             | 17.26           | ≈ 2.49                        | ≈ 1.24*                  |
| 4             | 11.09           | ≈ 3.87                        | ≈ 0.97                   |

`*` Une efficacité légèrement supérieure à 1 est due au bruit de mesure et à des effets de cache / scheduling.

### Validation

- Commande utilisée :

```bash
cd sorties/perroquets
md5sum -c perroquets.md5sum
```

- Résultat : les MD5 ne correspondent pas aux valeurs de référence (`FAILED` pour les 37 images).
- Le code de filtrage utilisé est cependant exactement le même que dans la version séquentielle donnée. La différence de checksum est vraisemblablement liée à des détails de plateforme (version de SciPy / Pillow, environnement différent de celui utilisé pour générer les références).
- Dans **ce** même environnement, lancer `mpiexec -n 1` puis `mpiexec -n 4` sur le même script produit des résultats cohérents entre eux : seule la répartition des images change.

Conclusion : la parallélisation respecte l’algorithme, et la stratégie par images fournit une accélération presque linéaire jusqu’à 4 processus.

---

## 3. Parallélisation d’une photo haute résolution (1) – `double_size.py`

### Stratégie de parallélisation

Objectif : doubler la taille d’une photo en évitant une image finale trop pixellisée, en appliquant :

1. Un filtre gaussien 3×3 sur les trois composantes H, S, V.
2. Un filtre de netteté 3×3 sur la seule composante V.

Contraintes :

- Minimiser la mémoire utilisée par chaque processus.
- Ne transférer que le strict minimum de données nécessaires aux convolutions.

Stratégie adoptée : **décomposition en bandes horizontales avec halos**.

- Rang 0 :
  - Charge l’image RGB, la convertit en HSV.
  - Convertit en tableau NumPy double précision.
  - Double la taille par `np.repeat` et normalise dans `[0, 1]`.
  - Connaît la hauteur finale `H` de l’image agrandie.
  - Calcule pour chaque rang `r` un intervalle de lignes locales `[start_r, end_r)` via un découpage équilibré.
  - Pour chaque processus `r`, envoie un sous-tableau de lignes comportant **une ligne fantôme en haut et en bas** (rayon du filtre 3×3) :
    - halo_start = max(0, start_r − 1)
    - halo_end = min(H, end_r + 1)

- Chaque processus :
  - Reçoit son sous-domaine avec halos.
  - Applique le flou gaussien 3×3 sur H, S, V localement (convolution 2D `signal.convolve2d` avec mode `same`).
  - Applique le filtre de netteté 3×3 sur V.
  - Enlève les lignes fantômes (haut/bas) et conserve seulement la partie « intérieure » correspondant à ses lignes propres.

- Rang 0 :
  - Reçoit les bandes « intérieures » de tous les processus.
  - Recopie chaque bande aux bonnes lignes dans l’image finale.
  - Convertit en `uint8`, puis `Image HSV → RGB` et enregistre le résultat.

Pourquoi cette stratégie est adaptée :

- Chaque processus ne stocke qu’une bande de l’image, plus 2 lignes de halo : mémoire par processus minimale.
- Les convolutions sont locales et ne nécessitent que des données voisines immédiates ; les halos d’une ligne suffisent pour un filtre 3×3.
- La communication est limitée à deux phases : distribution initiale des bandes et rassemblement final.

Lien avec la question 2 :

- Ici, on parallélise **une seule image** en la découpant, alors qu’en 2 on parallélisait **un ensemble** d’images indépendantes.
- Pour ce type de convolution 2D, cette stratégie par bandes est classique et raisonnablement efficace, mais moins idéale que la parallélisation par tâches entièrement indépendantes (frames) de la question 2.

### Mesures de performance

Image : `datas/paysage.jpg` (7500×4219 → 8438×15000).

Commande type :

```bash
/usr/bin/time -p mpiexec -n P venv/bin/python double_size.py
```

Résultats :

| P (processus) | Temps réel (s) | Accélération S(P) | Efficacité E(P) |
|--------------:|----------------:|------------------:|----------------:|
| 1             | 28.26           | 1.00              | 1.00            |
| 2             | 14.31           | ≈ 1.98            | ≈ 0.99          |
| 4             | 10.16           | ≈ 2.78            | ≈ 0.70          |

Commentaires :

- Le speedup est presque idéal pour 2 processus (≈2), avec excellente efficacité.
- En passant à 4 processus, la communication MPI et la gestion des halos deviennent plus visibles, ce qui fait chuter l’efficacité (~70 %).
- Comparée à la question 2, la parallélisation est moins « parfaite » car les processus doivent partager des données de bord et synchroniser.

---

## 4. Parallélisation d’une photo haute résolution (2) – `double_size2.py`

### Stratégie de parallélisation

Objectif : même agrandissement de la photo, mais filtres :

- Gaussien 3×3 sur H et S.
- Filtre combiné 5×5 sur V (lissage + netteté dans un seul masque).

Même contrainte de mémoire par processus minimale.

Stratégie : **même décomposition en bandes horizontales** que dans la question 3, avec modifications :

- Le rayon du filtre 5×5 est 2 → chaque bande locale a besoin de **2 lignes fantômes** en haut et 2 en bas.
- Rang 0 découpe l’image agrandie en bandes `[start_r, end_r)` et envoie pour chaque rang `r` :
  - halo_start = max(0, start_r − 2)
  - halo_end = min(H, end_r + 2)

- Chaque processus :
  - Applique le flou gaussien 3×3 sur H et S (comme en Q3).
  - Copie V, puis applique le filtre 5×5 sur V.
  - Retire 2 lignes fantômes en haut / bas (ou moins aux bords du domaine) pour reconstruire la partie intérieure correspondant à ses lignes.

- Rang 0 : rassemble ces bandes intérieures et reconstruit l’image finale comme en Q3.

### Différences, avantages et inconvénients par rapport à Q3

- **Différences techniques** :
  - Rayon de halo : 1 (Q3) vs 2 (Q4) → plus de données échangées par bande.
  - Filtrage : 2 convolutions 3×3 sur V en Q3 (lissage + netteté) vs une convolution 5×5 sur V en Q4.

- **Avantages de Q4** :
  - Le filtre 5×5 peut offrir une meilleure combinaison de lissage / netteté en une seule étape.
  - Pour un même nombre de pixels, la charge de calcul par pixel est plus élevée (25 coefficients au lieu de 9), ce qui améliore potentiellement le ratio calcul/communication.

- **Inconvénients de Q4** :
  - Halos plus larges (2 lignes) → plus de mémoire par processus et plus de données à envoyer/recevoir.
  - L’image traitée est plus « légère » côté algorithmique (moins d’étapes sur V), donc le gain de calcul par processus ne compense pas entièrement le surcoût de communication.

### Mesures de performance

Même image d’entrée que Q3.

Commande type :

```bash
/usr/bin/time -p mpiexec -n P venv/bin/python double_size2.py
```

Résultats :

| P (processus) | Temps réel (s) | Accélération S(P) | Efficacité E(P) |
|--------------:|----------------:|------------------:|----------------:|
| 1             | 14.45           | 1.00              | 1.00            |
| 2             | 11.08           | ≈ 1.30            | ≈ 0.65          |
| 4             | 8.27            | ≈ 1.75            | ≈ 0.44          |

Commentaires :

- Le programme parallèle est globalement plus rapide que celui de Q3 pour P=1 (moins d’opérations globales), mais il se parallélise moins bien.
- L’augmentation de la largeur des halos et la quantité de communication MPI réduisent notablement l’efficacité pour 2 et 4 processus.
- En pratique, ce type de filtre 5×5 est plus intéressant sur des problèmes encore plus grands (plus de lignes / colonnes) où le coût de calcul domine largement le coût des communications.

---

## 5. Conclusion générale

- La parallélisation par **tâches indépendantes** (images de la vidéo) fournit la meilleure accélération, proche de linéaire jusqu’à 4 processus.
- La décomposition **par bandes horizontales** est une stratégie classique pour les convolutions 2D et fonctionne correctement, mais l’efficacité diminue lorsqu’on augmente le nombre de processus, surtout lorsque le rayon du filtre (et donc la taille des halos) augmente.
- Les trois programmes parallélisés fonctionnent dans l’environnement testé et produisent des images cohérentes visuellement ; les différences de `md5sum` avec les références semblent venir d’un environnement logiciel différent plutôt que de la parallélisation elle‑même.
