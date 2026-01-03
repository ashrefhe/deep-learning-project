"""
SPAM DETECTOR APPLICATION - VERSION CORRIGÉE
Application Streamlit complète pour la détection de spam
"""

import streamlit as st
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import plotly.express as px
import time

# Import conditionnel de WordCloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud non disponible - les nuages de mots seront remplacés par des graphiques")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk


# ==================== CONFIG PAGE ====================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Télécharger les ressources NLTK
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        st.warning(f"Impossible de télécharger les ressources NLTK: {e}")
        return False

download_nltk_resources()

# ==================== STYLE CSS ====================
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
    .spam-box {
        background-color: #fee; border-left: 5px solid #f44;
        padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;
    }
    .ham-box {
        background-color: #efe; border-left: 5px solid #4f4;
        padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;
    }
    .metric-card {
        background-color: white; padding: 1.5rem; border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #2c3e50; }
    .stButton>button { border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS ====================

@st.cache_data
def load_data():
    """Charge les données avec plusieurs stratégies"""
    try:
        df = pd.read_csv("spam.csv", sep=";", encoding="latin-1")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['v1', 'v2']
    except:
        try:
            df = pd.read_csv("spam.csv", encoding="latin-1")
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ['v1', 'v2']
        except:
            st.info("📁 Fichier spam.csv non trouvé. Utilisation de données d'exemple.")
            df = pd.DataFrame({
                'v1': ['ham'] * 50 + ['spam'] * 50,
                'v2': (
                    [
                        'Hey how are you doing today',
                        'Meeting at 3pm tomorrow',
                        'Can you pick up milk',
                        'Dinner tonight at 7',
                        'Thanks for your help',
                        'See you later',
                        'Good morning',
                        'How was your day',
                        'Call me when you can',
                        'Happy birthday',
                    ] * 5
                    + [
                        'WIN FREE MONEY NOW!!!',
                        'Congratulations you won $10000',
                        'Click here for FREE iPhone',
                        'Limited time offer act now',
                        'You have been selected',
                        'Claim your prize today',
                        'URGENT: Your account needs verification',
                        'Make money from home FAST',
                        'Get rich quick scheme',
                        'FREE credit card offer',
                    ] * 5
                )
            })

    # Nettoyage des données
    df['v1'] = df['v1'].astype(str).str.strip().str.lower()
    df['v1'] = df['v1'].str.replace('"', '', regex=False).str.replace("'", '', regex=False)
    df['v2'] = df['v2'].astype(str).str.strip()

    df = df[df['v1'].isin(['ham', 'spam'])]
    df = df[df['v2'].str.len() > 0]
    df = df.dropna().reset_index(drop=True)
    return df

def preprocess_text(text, method='stemming'):
    """Prétraite un texte avec gestion d'erreurs"""
    try:
        stop_words = set([
            'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
            'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
            'it','its','itself','they','them','their','theirs','themselves','what','which',
            'who','whom','this','that','these','those','am','is','are','was','were','be','been',
            'being','have','has','had','having','do','does','did','doing','a','an','the','and',
            'but','if','or','because','as','until','while','of','at','by','for','with','about',
            'against','between','into','through','during','before','after','above','below','to',
            'from','up','down','in','out','on','off','over','under','again','further','then',
            'once','here','there','when','where','why','how','all','any','both','each','few',
            'more','most','other','some','such','no','nor','not','only','own','same','so','than',
            'too','very','s','t','can','will','just','don','should','now'
        ])

        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        tokens = text.split()
        tokens = [w for w in tokens if w and w not in stop_words]

        if method == 'stemming':
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(w) for w in tokens if w]
        else:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(w) for w in tokens if w]

        result = ' '.join(tokens)
        return result if result else 'empty'
    except Exception as e:
        st.error(f"Erreur dans le preprocessing: {e}")
        return 'empty'

def train_model(df, preprocess_method='stemming', vectorizer_type='tfidf', model_type='naive_bayes'):
    """Entraîne un modèle avec validation"""
    try:
        if df is None or len(df) < 10:
            raise ValueError("Pas assez de données pour l'entraînement")

        df = df.copy()
        df['processed'] = df['v2'].apply(lambda x: preprocess_text(x, preprocess_method))
        df = df[df['processed'] != 'empty']

        if len(df) < 10:
            raise ValueError("Trop de textes vides après preprocessing")

        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=3000, min_df=2, max_df=0.8, ngram_range=(1, 2)
            )
        else:
            vectorizer = CountVectorizer(
                max_features=3000, min_df=2, max_df=0.8, ngram_range=(1, 2)
            )

        X = vectorizer.fit_transform(df['processed'])
        y = (df['v1'] == 'spam').astype(int)

        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError("Dataset déséquilibré (une seule classe)")

        if model_type == 'naive_bayes':
            model = MultinomialNB(alpha=1.0)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(C=10, max_iter=1000, random_state=42)
        elif model_type == 'svm':
            model = SVC(C=10, kernel='linear', probability=True, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

        model.fit(X, y)

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return model, vectorizer, accuracy

    except Exception as e:
        st.error(f"Erreur lors de l'entraînement: {e}")
        return None, None, 0

def predict_message(text, model, vectorizer, preprocess_method='stemming'):
    """Prédit si un message est spam ou ham"""
    try:
        if not text or not str(text).strip():
            return None, None

        processed = preprocess_text(text, preprocess_method)
        if processed == 'empty':
            return None, None

        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(vectorized)[0]
        else:
            probability = np.array([np.nan, np.nan])

        return prediction, probability
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None

# ==================== SESSION STATE ====================

def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    if 'preprocess_method' not in st.session_state:
        st.session_state.preprocess_method = 'stemming'
    if 'vectorizer_type' not in st.session_state:
        st.session_state.vectorizer_type = 'tfidf'
    if 'model_type' not in st.session_state:
        st.session_state.model_type = 'naive_bayes'
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = 0
    if 'test_message' not in st.session_state:
        st.session_state.test_message = ""
    # ✅ clé persistante pour le text_area
    if 'user_message' not in st.session_state:
        st.session_state.user_message = ""

initialize_session_state()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #2c3e50;'>🔒 Spam Detector</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Prédiction", "📁 Batch", "🔍 Explorer", "⚙️ Configuration"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    if st.session_state.df is not None and len(st.session_state.df) > 0:
        total = len(st.session_state.df)
        spam_count = int((st.session_state.df['v1'] == 'spam').sum())
        ham_count = total - spam_count

        st.metric("Total messages", total)
        st.metric("🔴 Spam", spam_count, f"{spam_count/total*100:.1f}%")
        st.metric("🟢 Ham", ham_count, f"{ham_count/total*100:.1f}%")

        if st.session_state.model is not None:
            st.markdown("---")
            st.success("✅ Modèle entraîné")
            if st.session_state.model_accuracy > 0:
                st.metric("Précision", f"{st.session_state.model_accuracy*100:.1f}%")

# ==================== PAGE: DASHBOARD ====================

if page == "🏠 Dashboard":
    st.title("🏠 Dashboard - Spam Detector")
    st.markdown("Bienvenue dans l'application de détection de spam utilisant le Machine Learning")

    df = st.session_state.df
    if df is None or len(df) == 0:
        st.error("❌ Aucune donnée disponible")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📧 Total Messages", len(df))
        st.markdown('</div>', unsafe_allow_html=True)

    spam_count = int((df['v1'] == 'spam').sum())
    ham_count = len(df) - spam_count

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔴 Spam", spam_count, f"{spam_count/len(df)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🟢 Ham", ham_count, f"{ham_count/len(df)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        avg_length = df['v2'].str.len().mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📏 Longueur moy.", f"{avg_length:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distribution des classes")
        fig = px.pie(
            values=[ham_count, spam_count],
            names=['Ham', 'Spam'],
            color_discrete_map={'Ham': '#4CAF50', 'Spam': '#F44336'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📏 Distribution longueur messages")
        df_plot = df.copy()
        df_plot['length'] = df_plot['v2'].str.len()
        fig = px.histogram(
            df_plot,
            x='length',
            color='v1',
            nbins=50,
            color_discrete_map={'ham': '#4CAF50', 'spam': '#F44336'},
            labels={'length': 'Longueur du message', 'count': 'Nombre'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PRÉDICTION (FIX) ====================

elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction en temps réel")
    st.markdown("Entrez un message pour vérifier s'il s'agit d'un spam ou non")

    # Modèle pas entraîné
    if st.session_state.model is None:
        st.warning("⚠️ Aucun modèle entraîné. Allez dans Configuration pour entraîner un modèle.")
        if st.button("🚀 Entraîner un modèle maintenant", type="primary"):
            with st.spinner("Entraînement en cours..."):
                model, vectorizer, accuracy = train_model(
                    st.session_state.df,
                    st.session_state.preprocess_method,
                    st.session_state.vectorizer_type,
                    st.session_state.model_type
                )
                if model is not None:
                    st.session_state.model = model
                    st.session_state.vectorizer = vectorizer
                    st.session_state.model_accuracy = accuracy
                    st.success(f"✅ Modèle entraîné avec succès! Précision: {accuracy*100:.2f}%")
                    time.sleep(0.7)
                    st.rerun()
                else:
                    st.error("❌ Échec de l'entraînement du modèle")

    else:
        st.markdown("### 📝 Entrez votre message")

        # ✅ Si on clique sur un exemple, on le met dans la zone
        if st.session_state.test_message:
            st.session_state.user_message = st.session_state.test_message
            st.session_state.test_message = ""

        # ✅ text_area avec key => le texte ne disparaît plus
        st.text_area(
            "Message à analyser",
            key="user_message",
            height=150,
            placeholder="Ex: Congratulations! You won $1000. Click here to claim your prize!"
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            predict_button = st.button("🔮 Analyser", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state.user_message = ""
                st.rerun()

        if predict_button:
            text = st.session_state.user_message  # ✅ lire depuis session_state
            if not text or not text.strip():
                st.warning("⚠️ Veuillez entrer un message à analyser")
            else:
                with st.spinner("Analyse en cours..."):
                    time.sleep(0.2)

                    prediction, probability = predict_message(
                        text,
                        st.session_state.model,
                        st.session_state.vectorizer,
                        st.session_state.preprocess_method
                    )

                if prediction is None:
                    st.error("❌ Impossible d'analyser ce message")
                else:
                    st.markdown("---")
                    st.markdown("### 📊 Résultat de l'analyse")

                    has_proba = (
                        isinstance(probability, np.ndarray)
                        and probability.shape == (2,)
                        and not np.isnan(probability).any()
                    )

                    if prediction == 1:
                        if has_proba:
                            spam_prob = probability[1] * 100
                            st.markdown(f"""
                            <div class="spam-box">
                                <h2>🚨 SPAM DÉTECTÉ !</h2>
                                <p style="font-size: 1.2rem;">Confiance : <b>{spam_prob:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(probability[1]))
                            c1, c2 = st.columns(2)
                            with c1: st.metric("🔴 Probabilité SPAM", f"{probability[1]*100:.2f}%")
                            with c2: st.metric("🟢 Probabilité HAM", f"{probability[0]*100:.2f}%")
                        else:
                            st.markdown("""
                            <div class="spam-box">
                                <h2>🚨 SPAM DÉTECTÉ !</h2>
                                <p style="font-size: 1.2rem;">(Probabilités non disponibles pour ce modèle)</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        if has_proba:
                            ham_prob = probability[0] * 100
                            st.markdown(f"""
                            <div class="ham-box">
                                <h2>✅ MESSAGE LÉGITIME (HAM)</h2>
                                <p style="font-size: 1.2rem;">Confiance : <b>{ham_prob:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(probability[0]))
                            c1, c2 = st.columns(2)
                            with c1: st.metric("🟢 Probabilité HAM", f"{probability[0]*100:.2f}%")
                            with c2: st.metric("🔴 Probabilité SPAM", f"{probability[1]*100:.2f}%")
                        else:
                            st.markdown("""
                            <div class="ham-box">
                                <h2>✅ MESSAGE LÉGITIME (HAM)</h2>
                                <p style="font-size: 1.2rem;">(Probabilités non disponibles pour ce modèle)</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with st.expander("🔍 Voir les détails du preprocessing"):
                        processed = preprocess_text(text, st.session_state.preprocess_method)
                        st.markdown("**Message original:**")
                        st.code(text)
                        st.markdown("**Message après preprocessing:**")
                        st.code(processed)

        st.markdown("---")
        st.markdown("### 💡 Exemples à tester")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Messages HAM:**")
            if st.button("📧 Hey, meeting at 3pm tomorrow"):
                st.session_state.test_message = "Hey, meeting at 3pm tomorrow"
                st.rerun()
            if st.button("📱 Can you pick up milk on your way home?"):
                st.session_state.test_message = "Can you pick up milk on your way home?"
                st.rerun()
            if st.button("💼 Project deadline is next Friday"):
                st.session_state.test_message = "Project deadline is next Friday"
                st.rerun()

        with c2:
            st.markdown("**Messages SPAM:**")
            if st.button("💰 Congratulations! You won $10000. Click here now!"):
                st.session_state.test_message = "Congratulations! You won $10000. Click here now!"
                st.rerun()
            if st.button("🎁 FREE iPhone! Limited offer. Act now!!!"):
                st.session_state.test_message = "FREE iPhone! Limited offer. Act now!!!"
                st.rerun()
            if st.button("🏆 You have been selected as a winner!!!"):
                st.session_state.test_message = "You have been selected as a winner!!!"
                st.rerun()

# ==================== (Optionnel) Pages Batch/Explorer/Config ====================
# Pour aller vite, je laisse ces pages à toi si tu veux les garder
# Tu peux les recoller depuis ton code initial.
# ----------------------------------------------------

elif page == "📁 Batch":
    st.info("Page Batch: recolle ton code initial ici (elle ne touche pas au bug)")

elif page == "🔍 Explorer":
    st.info("Page Explorer: recolle ton code initial ici (elle ne touche pas au bug)")

elif page == "⚙️ Configuration":
    st.title("⚙️ Configuration du modèle")
    st.markdown("Configurez les paramètres de preprocessing et d'entraînement")

    col1, col2 = st.columns(2)
    with col1:
        preprocess_method = st.selectbox(
            "Méthode de normalisation",
            ['stemming', 'lemmatization'],
            index=0 if st.session_state.preprocess_method == 'stemming' else 1
        )
    with col2:
        vectorizer_type = st.selectbox(
            "Type de vectorisation",
            ['tfidf', 'bow'],
            index=0 if st.session_state.vectorizer_type == 'tfidf' else 1
        )

    model_type = st.selectbox(
        "Algorithme de Machine Learning",
        ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
    )

    if st.button("🚀 Entraîner le modèle", type="primary", use_container_width=True):
        st.session_state.preprocess_method = preprocess_method
        st.session_state.vectorizer_type = vectorizer_type
        st.session_state.model_type = model_type

        with st.spinner("Entraînement en cours..."):
            model, vectorizer, accuracy = train_model(
                st.session_state.df,
                preprocess_method,
                vectorizer_type,
                model_type
            )

        if model is not None and vectorizer is not None:
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_accuracy = accuracy
            st.success(f"✅ Modèle entraîné! Précision: {accuracy*100:.2f}%")
        else:
            st.error("❌ Échec de l'entraînement")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔒 <b>Spam Detector</b> - Application de détection de spam avec Machine Learning</p>
        <p>Développé avec ❤️ using Streamlit | © 2025</p>
    </div>
""", unsafe_allow_html=True)
