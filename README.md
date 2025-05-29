# AG_ENDOS_AI_Analyser

# Installation et Configuration du Serveur

Ce guide vous aide à configurer un environnement virtuel Python 3.11, à installer les dépendances requises (`PyTorch`, `NumPy`, `Flask`, `Pillow`, `matplotlib`) et à exécuter le projet.

## Prérequis

- Python 3.11 doit être installé sur votre système. Vous pouvez le télécharger [ici](https://www.python.org/downloads/).
- `pip`, le gestionnaire de paquets Python, doit être disponible. Il est inclus par défaut avec Python 3.11.
- `virtualenv` pour créer des environnements virtuels (optionnel si vous utilisez déjà `venv`).

## Étapes d'installation

1. **Vérifiez votre version de Python :**

   ```bash
   python --version
   ```

   Assurez-vous que la version affichée est 3.11.x.

2. **Cloner le projet :**

   ```bash
   git clone https://github.com/youssef-ibnouali/AG_ENDOS_AI_Analyser.git
   cd AG_ENDOS_AI_Analyser
   ```

3. **Créez un environnement virtuel :**

   Dans le répertoire de votre projet, exécutez la commande suivante :

   ```bash
   python -m venv env
   ```

   Cela crée un environnement virtuel nommé `env`.

4. **Activez l'environnement virtuel :**

   - Sur **Windows** :
     ```bash
     env\Scripts\activate
     ```
   - Sur **Linux/MacOS** :
     ```bash
     source env/bin/activate
     ```

   Une fois activé, vous verrez un préfixe `(env)` dans votre terminal.

5. **Installez les dépendances nécessaires :**

   Exécutez la commande suivante pour installer les bibliothèques :

   ```bash
   pip install -r requirements.txt
   ```
