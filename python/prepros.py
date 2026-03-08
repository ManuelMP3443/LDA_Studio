import os
import re
from collections import defaultdict
import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Descargas NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Imports Opcionales
try:
    import docx
except ImportError:
    docx = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

class Prepros:
    def __init__(self):
        try:
            self.nlp_es = spacy.load("es_core_news_sm")
        except OSError:
            # --- FIX PARA PYINSTALLER ---
            import sys
            base_path = sys._MEIPASS if hasattr(sys, "_MEIPASS") else ""
            modelo_path = os.path.join(base_path, "es_core_news_sm")
            print("Cargando modelo desde:", modelo_path)
            self.nlp_es = spacy.load(modelo_path)

        self.lemmatizer_en = WordNetLemmatizer()
        self.stop_es = set(stopwords.words("spanish"))
        self.stop_en = set(stopwords.words("english"))

    def _limpiar_y_procesar(self, texto, idioma):
        if not isinstance(texto, str): return []
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", texto)
        
        if idioma and idioma.startswith("en"):
            texto = re.sub(r"[^a-z\s]", "", texto)
            tokens = nltk.word_tokenize(texto)
            stopw = self.stop_en
            lemmatizer = lambda x: [self.lemmatizer_en.lemmatize(t) for t in x]
        else:
            texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
            tokens = nltk.word_tokenize(texto)
            stopw = self.stop_es
            # Lematización por lotes para velocidad
            lemmatizer = lambda x: [t.lemma_ for t in self.nlp_es(" ".join(x))]

        tokens = [t for t in tokens if len(t) > 2 and t not in stopw]
        return lemmatizer(tokens)

    def _crear_vocabulario(self, corpus, min_freq=1, max_ratio=1.0):
        frecuencias = defaultdict(int)
        for doc in corpus:
            for w in doc: frecuencias[w] += 1
            
        D = len(corpus)
        max_doc_freq = D * max_ratio if D > 0 else 0
        vocab = {}
        idx = 0
        for w, f in frecuencias.items():
            if f >= min_freq:
                vocab[w] = idx
                idx += 1
        return vocab

    def _mapear_corpus(self, corpus, vocab):
        return [[vocab[w] for w in doc if w in vocab] for doc in corpus]

    def _leer_archivo_texto(self, ruta):
        """Lee un archivo específico ignorando errores."""
        ext = os.path.splitext(ruta)[1].lower()
        texto = ""
        try:
            if ext == ".txt":
                try:
                    with open(ruta, "r", encoding="utf-8") as f: texto = f.read()
                except UnicodeDecodeError:
                    with open(ruta, "r", encoding="latin-1") as f: texto = f.read()
            elif ext == ".docx" and docx:
                doc = docx.Document(ruta)
                texto = "\n".join([p.text for p in doc.paragraphs])
            elif ext == ".pdf" and PyPDF2:
                # Solo entra aquí si se seleccionas explícitamente un PDF
                with open(ruta, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texto = "\n".join([p.extract_text() for p in reader.pages])
        except Exception as e:
            print(f"⚠️ Error leyendo {os.path.basename(ruta)}: {e}")
        return texto

    def _guardar_archivos(self, salida_dir, K, V, D, alpha, beta, iteraciones, vocab, corpus):
        os.makedirs(salida_dir, exist_ok=True)
        with open(os.path.join(salida_dir, "config.txt"), "w") as f:
            f.write(f"K={K}\nV={V}\nD={D}\nALPHA={alpha}\nBETA={beta}\nITERACIONES={iteraciones}\n")
        
        inv_vocab = {v: k for k, v in vocab.items()}
        with open(os.path.join(salida_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            for i in range(V): f.write(f"{i},{inv_vocab.get(i,'')}\n")
            
        with open(os.path.join(salida_dir, "corpus.txt"), "w") as f:
            for doc in corpus: f.write(" ".join(map(str, doc)) + "\n")
            
        return {"config": os.path.join(salida_dir, "config.txt")}

    def run(self, ruta_entrada, K_TEMAS, Beta, Idioma="spanish", flag_csv=True, 
            columna_csv=None, alfa=None, ITERACIONES=1000):
        
        # Parsear K
        if isinstance(K_TEMAS, str):
            lista_k = [int(k) for k in re.split(r"[ ,]+", K_TEMAS.strip()) if k.isdigit()]
        else:
            lista_k = K_TEMAS if isinstance(K_TEMAS, list) else [K_TEMAS]

        textos = []
        base_dir = ""

        if flag_csv:
            # CSV Único
            if isinstance(ruta_entrada, list): ruta_entrada = ruta_entrada[0]
            try:
                df = pl.read_csv(ruta_entrada)
                textos = df[columna_csv].to_list()
                base_dir = os.path.dirname(os.path.abspath(ruta_entrada))
            except Exception as e:
                return {"success": False, "error": f"Error CSV: {e}"}
        else:
            # Lista de Archivos (TXT/DOCX/PDF)
            if not ruta_entrada:
                 return {"success": False, "error": "Lista de archivos vacía."}
            
            # Tomamos el directorio del primer archivo para guardar la salida
            base_dir = os.path.dirname(os.path.abspath(ruta_entrada[0]))

            for ruta in ruta_entrada:
                # Solo lee lo que el usuario seleccionó
                txt = self._leer_archivo_texto(ruta)
                if txt.strip(): textos.append(txt)

        if not textos:
            return {"success": False, "error": "No se pudo extraer texto válido."}

        # Procesamiento NLP
        corpus_limpio = []
        for t in textos:
            tokens = self._limpiar_y_procesar(t, Idioma)
            if tokens: corpus_limpio.append(tokens)

        # Vocabulario (min_freq=1 para pruebas pequeñas)
        vocab = self._crear_vocabulario(corpus_limpio, min_freq=1, max_ratio=1.0)
        V = len(vocab)
        if V == 0: return {"success": False, "error": "Vocabulario vacío."}

        corpus_mapeado = self._mapear_corpus(corpus_limpio, vocab)
        D = len(corpus_mapeado)

        # Generar Salidas
        salida_dir = os.path.join(base_dir, "salida")
        resultados_k = {}
        
        for k in lista_k:
            path_k = os.path.join(salida_dir, f"K_{k}")
            alpha_val = alfa if alfa else 50.0/k
            res = self._guardar_archivos(path_k, k, V, D, alpha_val, Beta, ITERACIONES, vocab, corpus_mapeado)
            resultados_k[k] = {"path": path_k, "config": res["config"]}

        return {
            "success": True, "V": V, "D": D, 
            "base_salida_dir": salida_dir, 
            "resultados_por_k": resultados_k
        }