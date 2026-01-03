# Spam Detector 📨

**Détection de Spam avec NLP et Machine Learning**  
Une application intelligente pour détecter les spams (SMS & emails) grâce au NLP et au Machine Learning. Ce projet inclut des notebooks Jupyter pour l’exploration et l’entraînement des modèles, ainsi qu’une application web Streamlit pour une utilisation interactive.

---

## 🎬 Démo Vidéo
Regardez la démonstration complète ici : [YouTube](#)

---

## 📖 Description
Ce projet repose sur un dataset de messages SMS (ham vs. spam) et utilise :  
- **Prétraitement du texte** : nettoyage, stopwords, ponctuation, stemming, lemmatisation.  
- **Vectorisation** : TF-IDF ou Bag-of-Words.  
- **Classification ML** : Naive Bayes, Logistic Regression, SVM, Random Forest.  
- **Deep Learning** : LSTM avec Keras pour des séquences complexes.  

Les notebooks (`v9.ipynb`, `v10.ipynb`, `Travail_DL_TestNotebook.ipynb`) montrent toute l’évolution du projet, de l’analyse exploratoire à l’entraînement de modèles profonds.  
L’application Streamlit (`app.py`) permet une détection en temps réel, le traitement batch et des visualisations interactives.

---

## 💡 Points forts
- Prétraitement robuste du texte.  
- Balancing des données pour éviter le biais.  
- Visualisations interactives : histogrammes, nuages de mots, graphiques dynamiques.  
- Interface utilisateur intuitive avec Streamlit.

---

## ⚡ Fonctionnalités
1. **Prédiction en temps réel** : testez un message individuel.  
2. **Analyse batch** : analysez plusieurs messages via CSV.  
3. **Exploration des données** : distribution des classes, longueur des messages, nuages de mots.  
4. **Configuration du modèle** : ajustez le prétraitement, la vectorisation et l’algorithme ML.  
5. **Dashboard** : statistiques et graphiques interactifs.  
6. **Support multilingue** : français/anglais.

---

## 🛠 Technologies
- **Langage** : Python 3.12  
- **NLP / ML** : NLTK, Scikit-learn, Pandas, NumPy  
- **Visualisation** : Matplotlib, Seaborn, Plotly, WordCloud  
- **Deep Learning** : Keras (LSTM, embeddings)  
- **Interface** : Streamlit  
- **Environnement** : Jupyter Notebooks  

---

## ⚙️ Installation
1. Clonez le dépôt :  
```bash
git clone https://github.com/votre-utilisateur/spam-detector.git
cd spam-detector
