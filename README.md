# Deep Learning MLOps – Fruits 360 Classification

## Objectif du projet
Ce projet vise à illustrer le cycle de vie complet d’un modèle de Deep Learning pour la classification d’images, du ré-entraînement à la mise en production via une API conteneurisée.

---

## 1. Choix du Dataset
- **Dataset utilisé** : [Fruits 360 (Kaggle)](https://www.kaggle.com/datasets/moltean/fruits)
- **Justification** :
  - Taille raisonnable (environ 1 Go)
  - Problème multi-classes (206 types de fruits)
  - Images bien étiquetées, format standard

---

## 2. Préparation de l’Environnement
### Problèmes rencontrés
- **Gestion des dépendances** :
  - Sur Mac (Apple Silicon), l’installation de TensorFlow/Keras peut poser problème (versions spécifiques, incompatibilités, etc.).
  - L’environnement virtuel Python (`venv`) a généré des erreurs liées à la gestion externe des paquets (PEP 668, Homebrew).
- **Solution** :
  - Abandon de l’environnement virtuel local au profit d’une image Docker pour garantir la portabilité et la reproductibilité.
  - Utilisation de `tensorflow` version CPU dans Docker (compatible partout), mais recommandation de `tensorflow-macos`/`tensorflow-metal` pour l’entraînement local sur Mac M1/M2/M3/M4.

---

## 3. Librairies et Outils
- **Deep Learning** : TensorFlow 2.15, Keras 2.15
- **API** : FastAPI, Uvicorn
- **Gestion des images** : Pillow
- **Gestion des données** : Pandas, Numpy, Scikit-learn
- **Téléchargement dataset** : Kaggle API
- **Conteneurisation** : Docker

---

## 4. Structure du Projet
```
DeepLearning_MLops/
├── fruits360_data/
│   ├── Training/
│   └── Test/
├── train_resnet50.py
├── requirements.txt
├── Dockerfile
├── README.md
```

---

## 5. Préparation des Données
- Téléchargement manuel du dataset depuis Kaggle (problèmes d’API résolus par copie manuelle).
- Copie des dossiers `Training` et `Test` dans `fruits360_data/`.

---

## 6. Entraînement du Modèle
- **Modèle choisi** : ResNet50 pré-entraîné sur ImageNet (transfer learning, top layer adaptée).
- **Pourquoi ResNet50 ?**
  - Excellente performance sur la classification d’images.
  - Support natif dans Keras/TensorFlow.
  - Compatible avec l’accélération matérielle Apple Silicon (en local).
- **Paramètres d’entraînement** :
  - Image size : 100x100
  - Batch size : 32
  - Epochs : 5 (pour la démonstration, à augmenter pour de meilleures performances)
  - Optimizer : Adam (lr=1e-4)
  - Augmentation de données : rotation, shift, zoom, flip
- **Résultats obtenus** :
  - Accuracy finale (val) : ~7.8% (5 epochs, modèle non fine-tuné, uniquement la top layer entraînée)
  - À augmenter pour de meilleures performances (plus d’epochs, fine-tuning possible)
- **Sauvegarde** :
  - Modèle sauvegardé sous `resnet50_fruits360.h5`

---

## 7. Conteneurisation avec Docker
- **Pourquoi Docker ?**
  - Résout les problèmes de dépendances et d’environnement.
  - Permet de reproduire l’entraînement et l’inférence sur n’importe quelle machine.
- **Fichiers clés** :
  - `Dockerfile` :
    ```dockerfile
    FROM python:3.9-slim-buster
    WORKDIR /app
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 8000
    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
  - `requirements.txt` :
    ```
    tensorflow==2.15.0
    keras==2.15.0
    kaggle==1.7.4.5
    fastapi==0.110.0
    uvicorn[standard]==0.29.0
    pillow==10.3.0
    python-multipart==0.0.9
    scikit-learn==1.4.2
    numpy==1.26.4
    pandas==2.2.2
    ```
- **Construction de l’image** :
  ```bash
  docker build -t fruits360-mlops:latest .
  ```
- **Lancement de l’entraînement** :
  ```bash
  docker run --rm -v $(pwd):/app -w /app fruits360-mlops:latest python train_resnet50.py
  ```

---

## 8. Développement de l’API d’inférence avec FastAPI

- **Pourquoi FastAPI ?**
  - FastAPI est un framework moderne, rapide et asynchrone pour créer des APIs REST en Python.
  - Il offre une documentation automatique (Swagger/OpenAPI), une gestion native de la validation des données, et des performances élevées.
  - Idéal pour le déploiement de modèles de machine learning en production grâce à sa simplicité et sa rapidité.

- **Structure de l’API** :
  - Le code de l’API se trouve dans le dossier `API/`.
  - L’API charge le modèle entraîné (`resnet50_fruits360.h5`) au démarrage.
  - **Un seul endpoint POST `/predict`** :
    - **URL** : `POST /predict`
    - **Description** : Prend en entrée une image (JPG ou PNG) et retourne la classe prédite et la confiance.
    - **Payload attendu** :
      - Form-data, champ `file` contenant l’image à prédire.
    - **Exemple de requête (curl)** :
      ```bash
      curl -X POST "http://localhost:8000/predict" -F "file=@chemin/vers/une_image.jpg"
      ```
    - **Exemple de réponse** :
      ```json
      {
        "predicted_class": "Apple Red 1",
        "confidence": 0.98
      }
      ```
    - **Sécurité & Robustesse** :
      - Seuls les fichiers JPEG/PNG sont acceptés.
      - Gestion des erreurs et des formats non supportés.
      - Documentation automatique désactivée pour limiter l’exposition (`docs_url=None, redoc_url=None, openapi_url=None`).

- **Lancement de l’API dans Docker** :
  - Le `Dockerfile` a été adapté pour lancer directement l’API avec Uvicorn :
    ```dockerfile
    CMD ["uvicorn", "API.api:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
  - Pour lancer l’API seule :
    ```bash
    docker run --rm -v $(pwd):/app -w /app -p 8000:8000 fruits360-mlops:latest
    ```
  - Pour lancer avec le front Streamlit, utiliser Docker Compose (voir section suivante).

---

## 9. Interface Utilisateur avec Streamlit

- **Pourquoi Streamlit ?**
  - Permet de créer rapidement une interface web interactive pour tester le modèle sans écrire de code front complexe.
  - Supporte le drag-and-drop, l’affichage d’images, et l’intégration facile avec une API REST.
  - Idéal pour la démo, la validation métier ou le prototypage.

- **Fonctionnalités** :
  - Drag-and-drop pour uploader une image (jpg/png).
  - Affichage de l’image uploadée.
  - Envoi automatique de la requête à l’API FastAPI `/predict`.
  - Affichage de la classe prédite et du score de confiance.

- **Intégration avec Docker Compose** :
  - Le front Streamlit et l’API FastAPI tournent dans des conteneurs séparés mais sur le même réseau Docker.
  - Utilisation de la variable d’environnement `API_URL` pour que Streamlit communique avec l’API, que ce soit en local ou dans Docker Compose.

- **Lancement** :
  ```bash
  docker compose up --build
  ```
  - Accès à l’interface : http://localhost:8501
  - Accès à l’API : http://localhost:8000 (endpoint POST /predict)

- **Problèmes rencontrés et solutions** :
  - **Erreur de connexion entre Streamlit et l’API** :
    - Cause : Les deux services n’étaient pas sur le même réseau, ou l’API n’était pas démarrée.
    - Solution : Utilisation de Docker Compose pour orchestrer les deux services et passage de l’URL de l’API via une variable d’environnement.
  - **Port 8501 déjà utilisé** :
    - Cause : Ancien conteneur ou instance Streamlit encore en cours.
    - Solution : Arrêt de tous les conteneurs utilisant l’image, puis relance de Docker Compose.
  - **Module API non trouvé par Uvicorn** :
    - Cause : Absence de `__init__.py` dans le dossier `API/`.
    - Solution : Création du fichier pour que le dossier soit reconnu comme package Python.
  - **404 sur http://localhost:8000/** :
    - Cause : L’API n’expose que le endpoint POST `/predict` (comportement normal).

- **Bonnes pratiques** :
  - Toujours vérifier la structure du projet et la présence des fichiers nécessaires (`__init__.py`, modèle, etc.).
  - Utiliser des variables d’environnement pour la configuration dynamique entre services.
  - Orchestrer les services avec Docker Compose pour éviter les problèmes de réseau et de port.

---

## 10. Conseils et bonnes pratiques
- Toujours vérifier la compatibilité des librairies avec votre matériel (notamment sur Mac Apple Silicon).
- Privilégier Docker pour la portabilité et la reproductibilité.
- Pour de meilleures performances, augmenter le nombre d’epochs et envisager le fine-tuning complet du modèle.
- Utiliser FastAPI pour des APIs performantes, bien documentées et faciles à maintenir.
- Utiliser Streamlit pour une interface utilisateur rapide et interactive.

---

## 11. Difficultés rencontrées et solutions
- **Problèmes d’installation de dépendances sur Mac (PEP 668, Homebrew)** :
  - Solution : Docker pour isoler l’environnement.
- **API Kaggle et gestion du fichier kaggle.json** :
  - Solution : téléchargement manuel du dataset.
- **Temps d’entraînement** :
  - Pour un vrai projet, prévoir plus d’epochs et/ou l’utilisation d’un GPU.
- **Déploiement de l’API** :
  - FastAPI a permis une intégration rapide et une documentation automatique, facilitant les tests et l’intégration continue.
- **Connexion front/API et orchestration** :
  - Solution : Docker Compose, variable d’environnement API_URL, gestion des ports et des dépendances de service.
- **Structure de projet et import Python** :
  - Solution : Ajout de `__init__.py` dans le dossier API.

---

## 12. Références
- [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 13. Partage et déploiement de l’image Docker

- **Publication sur Docker Hub** :
  - L’image Docker du projet est disponible publiquement ici :
    ```
    docker pull kimuntu/fruits360-mlops:latest
    ```
  - [Lien Docker Hub](https://hub.docker.com/r/kimuntu/fruits360-mlops)

- **Étapes pour publier l’image** :
  1. Connexion à Docker Hub :
     ```bash
     docker login -u <votre_nom_utilisateur>
     ```
  2. Taguer l’image :
     ```bash
     docker tag fruits360-mlops:latest <votre_nom_utilisateur>/fruits360-mlops:latest
     ```
  3. Pousser l’image :
     ```bash
     docker push <votre_nom_utilisateur>/fruits360-mlops:latest
     ```

- **Attention à l’architecture de l’image Docker** :
  - Par défaut, l’image a été construite sur une machine Mac Apple Silicon (architecture `linux/arm64`).
  - **Si vous souhaitez utiliser l’image sur une plateforme cloud qui attend du `linux/amd64` (ex : RunPod, certains services CI/CD, ou des serveurs x86 classiques)** :
    - Il faut reconstruire l’image en forçant l’architecture :
      ```bash
      docker buildx build --platform linux/amd64 -t <votre_nom_utilisateur>/fruits360-mlops:latest .
      docker push <votre_nom_utilisateur>/fruits360-mlops:latest
      ```
    - Sinon, l’image risque de ne pas démarrer ou d’être incompatible.
  - **Sur RunPod (template custom)** :
    - Il n'est pas possible de choisir l'architecture du pod dans l'interface RunPod : les pods attendent des images Docker en `linux/amd64` (x86) par défaut.
    - **Il est donc impératif de builder et pousser l'image Docker en `linux/amd64` même si vous êtes sur Mac ARM** :
      ```bash
      docker buildx build --platform linux/amd64 -t kimuntu/fruits360-mlops:latest --push .
      ```
    - Sans cela, l'image ne démarrera pas sur RunPod (erreur d'architecture incompatible).

- **Pour ce projet** :
  - Nous allons directement déployer l’image sur une instance EC2 avec l’architecture par défaut `linux/arm64` (compatible avec les instances Graviton ou Mac M1/M2/M3/M4).
  - Ce choix garantit la compatibilité et la performance sur les machines ARM modernes, tout en restant portable sur d’autres plateformes si besoin (avec rebuild multi-arch).

- **Résumé des bonnes pratiques** :
  - Toujours vérifier l’architecture cible de votre plateforme cloud avant de déployer une image Docker.
  - Utiliser `docker buildx` pour générer des images multi-architecture si besoin de portabilité maximale.
  - Documenter clairement l’architecture de build et de déploiement dans vos projets MLOps.

---

*Projet réalisé sur MacOS (Apple Silicon M4 PRO), optimisé pour la portabilité, la reproductibilité et la rapidité de déploiement grâce à Docker, FastAPI et Streamlit.* 

# Déploiement sur EC2 AWS

## 1. Pré-requis
- Avoir un compte AWS et une instance EC2 Ubuntu (t2.medium ou plus recommandé)
- Ouvrir les ports nécessaires dans le groupe de sécurité :
  - 22 (SSH)
  - 80 (HTTP, si besoin)
  - 8501 (Streamlit)
  - 8000 (API)
- Avoir la clé SSH (`mlops.pem`) sur votre machine locale

## 2. Connexion à l’instance EC2
```bash
ssh -i ~/Downloads/mlops.pem ubuntu@<IP-EC2>
```
Remplacez `<IP-EC2>` par l’adresse publique de votre instance (ex : `ec2-54-208-26-189.compute-1.amazonaws.com`)

## 3. Installation des dépendances sur EC2
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg git
sudo apt install -y cloud-guest-utils # pour growpart si besoin
# Installer Docker (méthode officielle)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
# Installer Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
# Ajouter l’utilisateur au groupe docker
sudo usermod -aG docker $USER
# Déconnectez-vous puis reconnectez-vous pour activer le groupe docker
```

## 4. Cloner le projet et configurer Docker Compose
```bash
git clone <url-de-votre-repo>
cd DeepLearning_MLops
```
- Modifiez le fichier `docker-compose.yml` pour utiliser l’image Docker Hub :
```yaml
services:
  api:
    image: kimuntu/fruits360-mlops:latest
    ...
  streamlit:
    image: kimuntu/fruits360-mlops:latest
    ...
```

## 5. Lancer les services
```bash
docker-compose pull
docker-compose up -d
```

## 6. Accéder à l’application
- **Streamlit** : http://<IP-EC2>:8501
- **API** : http://<IP-EC2>:8000

## 7. Problèmes courants et solutions
- **Port non accessible** : Vérifiez le groupe de sécurité AWS (ports 8501, 8000 ouverts)
- **Erreur "no space left on device"** :
  - Libérez de l’espace (`docker system prune -a`, supprimez des fichiers inutiles)
  - Ou augmentez la taille du disque EBS (voir console AWS > Volumes > Modify Volume)
  - Redimensionnez la partition sur Ubuntu :
    ```bash
    sudo growpart /dev/xvda 1
    sudo resize2fs /dev/xvda1
    ```
- **Erreur de permission Docker** : Déconnectez-vous/reconnectez-vous après `usermod -aG docker $USER`
- **IP EC2 a changé** : Après un redémarrage, vérifiez la nouvelle IP publique dans la console AWS

## 8. Architecture locale vs cloud
- **Local** :
  - Build et test des images Docker sur votre machine (Mac/Windows/Linux)
  - Utilisez `docker-compose up --build` pour tester l’intégration
- **Cloud (EC2)** :
  - Utilisez des images multi-architecture (`buildx` pour `linux/amd64` et `linux/arm64`)
  - Privilégiez le pull d’images depuis Docker Hub pour rapidité et cohérence
  - Séparez bien les environnements de dev/test/prod

## 9. Bonnes pratiques et rigueur pour un bon développement
- Versionnez toutes les modifications (git)
- Documentez chaque étape et chaque problème rencontré dans le README
- Utilisez des images Docker légères et multi-arch
- Gardez votre code et vos dépendances à jour
- Sécurisez vos accès (SSH, ports, variables d’environnement)
- Automatisez le plus possible (scripts d’installation, CI/CD)
- Testez localement avant tout déploiement cloud
- Surveillez l’utilisation disque et mémoire sur EC2

---

**Pour toute question ou problème, consultez la documentation officielle AWS, Docker, ou ouvrez une issue sur le repo !** 