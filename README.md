##### \# Projet Budget Participatif - Les projets lauréats

##### 

##### Par \*\*Nizar Boussabat\*\*

##### 

##### \## 📊 Partie 1 : Analyse des projets

##### Ce projet s’appuie sur le dataset officiel des projets lauréats du Budget Participatif de Paris :

##### 

##### \- Dataset : \[bp\_projets\_gagnants](https://opendata.paris.fr/explore/dataset/bp\_projets\_gagnants/table/?disjunctive.thematique\&disjunctive.direction\_pilote\_projet\&disjunctive.echelle\_bp\&disjunctive.arrondissement\_projet\_gagnant\&disjunctive.avancement\_projet)

##### 

##### \- Analyse complète disponible sur Google Colab :

#####   \[Lien vers l’analyse](https://colab.research.google.com/drive/1lTeORkLjKeAwSlJVdLGZOxdLJl6ZvXLw#scrollTo=bDF1e2vGowbR)

#####   (Un fichier offline est également fourni)

##### 

##### \## 🤖 Partie 2 : Chatbot Citoyen

##### 

##### \### 1. Description

##### Ce projet est un \*\*chatbot citoyen\*\* conçu pour aider les habitants à :

##### \- Formuler des idées de projets locaux

##### \- Explorer les projets existants

##### \- Comparer les propositions

##### 

##### Il s’appuie sur des techniques de \*\*traitement du langage naturel (NLP)\*\* et une logique de \*\*détection d’intention\*\*.

##### 

##### \### 2. LLM utilisé

##### Le chatbot utilise \*\*Phi-3 :mini\*\*, un modèle de langage développé par \*\*Microsoft\*\*, optimisé pour être :

##### \- Léger et rapide

##### \- Facile à déployer en local ou embarqué

##### \- Efficace pour des projets citoyens et éducatifs

##### 

##### Phi-3 :mini est particulièrement adapté aux projets où la \*\*simplicité de déploiement\*\* et la \*\*performance\*\* sont essentielles.



##### \#### 3.Screenshots du fonctionnement du chatbot:

##### > Debut Chatbot:

##### !\[Simulation chatbot 1](Partie\_2\_Chatbot\_Citoyen/Screenshots/simulation\_chatbot1.png)

##### > L'utilisateur donne une description et le chatbot cherche des similaires dans le dataset:

##### !\[Simulation chatbot 2](Partie\_2\_Chatbot\_Citoyen/Screenshots/simulation\_chatbot2.png)

##### > Génération des suggestions des projets avec LLM:

##### !\[Simulation chatbot 2](Partie\_2\_Chatbot\_Citoyen/Screenshots/simulation\_chatbot3.png)

##### 

##### ---

##### 

##### \## 🚀 Lancer l’application avec Streamlit

##### 

##### \### Prérequis

##### \- Python 3.9+

##### \- Installation des dépendances :

##### ```bash

##### pip install -r requirements.txt

##### \### Démarrage de l'application

##### cd "Partie 2 Chatbot Citoyen"

##### streamlit run app.py



##### 

##### 

##### 

##### 

##### 

#####  

