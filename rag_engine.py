import os
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Garante o download das stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def extrair_texto_pdf(caminho_pdf):
    """Extrai todo o conteúdo de texto de um arquivo PDF."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")
    
    texto = ""
    with open(caminho_pdf, 'rb') as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        for pagina in leitor.pages:
            conteudo = pagina.extract_text()
            if conteudo:
                texto += conteudo + "\n"
    return texto

def tratar_texto(texto):
    """Limpa o texto removendo caracteres especiais e stopwords em português."""
    texto_limpo = re.sub(r'\W+', ' ', texto).lower()
    stopwords_pt = set(stopwords.words('portuguese'))
    palavras = [p for p in texto_limpo.split() if p not in stopwords_pt]
    return ' '.join(palavras)

def criar_indice_invertido(texto):
    """Cria índice vetorial FAISS usando vetorização TF-IDF."""
    vetorizador = TfidfVectorizer()
    vetores = vetorizador.fit_transform([texto])
    dimensao = vetores.shape[1]
    
    index = faiss.IndexFlatL2(dimensao)
    vetor_array = vetores.toarray().astype('float32')
    faiss.normalize_L2(vetor_array)
    index.add(vetor_array)
    
    return index, vetorizador
