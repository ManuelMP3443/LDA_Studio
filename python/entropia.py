# -*- coding: utf-8 -*-
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

class EvaluadorLDA:
    def __init__(self, ruta_base_salida):
        self.ruta_base = ruta_base_salida
        self.archivo_vocab = os.path.join(ruta_base_salida, "vocab.txt")
        self.archivo_corpus = os.path.join(ruta_base_salida, "corpus.txt")
        
        self.vocab_map = {}
        self.raw_docs = []
        self.todos_los_topicos = [] 
        self.img_path = os.path.join(self.ruta_base, 'grafico_log_entropia.png')

    def cargar_datos(self, k_ejemplo=None):
        print(f"--- Cargando datos base ---")
        # Búsqueda fallback si no están en la raíz
        if not os.path.exists(self.archivo_vocab) and k_ejemplo:
             self.archivo_vocab = os.path.join(self.ruta_base, f"K_{k_ejemplo}", "vocab.txt")
             self.archivo_corpus = os.path.join(self.ruta_base, f"K_{k_ejemplo}", "corpus.txt")

        if not os.path.exists(self.archivo_vocab) or not os.path.exists(self.archivo_corpus):
            return False

        try:
            with open(self.archivo_vocab, 'r', encoding='utf-8') as f:
                self.vocab_map = {}
                for i, line in enumerate(f):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        self.vocab_map[parts[1]] = i
            
            with open(self.archivo_corpus, 'r', encoding='utf-8') as f:
                self.raw_docs = [line.strip().split() for line in f]
                
            print(f"Documentos: {len(self.raw_docs)} | Vocabulario: {len(self.vocab_map)}")
            return True
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False

    def _calcular_score_topico_log_natural(self, phi_k):
        p = phi_k[phi_k > 0]
        return -np.sum(p * np.log(p))

    def procesar_modelo(self, k):
        nombre_carpeta = f"K_{k}" 
        ruta_carpeta = os.path.join(self.ruta_base, nombre_carpeta)
        
        path_n_kt = os.path.join(ruta_carpeta, 'n_kt.txt')
        path_n_mk = os.path.join(ruta_carpeta, 'n_mk.txt')
        
        if not os.path.exists(path_n_kt) or not os.path.exists(path_n_mk):
            return None

        try:
            n_kt = np.loadtxt(path_n_kt)
            n_mk = np.loadtxt(path_n_mk)
            
            # Correcciones de forma 
            if n_mk.ndim == 1:
                if n_mk.shape[0] == k: n_mk = n_mk.reshape(1, k)
                else: n_mk = n_mk.reshape(-1, 1)
            if n_kt.ndim == 1: n_kt = n_kt.reshape(1, -1)
            if n_kt.shape[0] != k and n_kt.shape[1] == k: n_kt = n_kt.T
                
        except Exception as e:
            print(f"Error al cargar matrices K={k}: {e}")
            return None

        # 1. Normalizar Theta y Phi (Probabilidades)
        phi = n_kt / (n_kt.sum(axis=1)[:, np.newaxis] + 1e-10)       # (K x V)
        theta = n_mk / (n_mk.sum(axis=1)[:, np.newaxis] + 1e-10)     # (D x K)

        # 2. Calcular métricas internas de tópicos para el Top 20 (Shannon)
        for i in range(phi.shape[0]):
            p = phi[i, :]
            p = p[p > 0]
            score = -np.sum(p * np.log(p)) # Shannon simple para el ranking
            self.todos_los_topicos.append((score, k, i))

        suma_log_probabilidades = 0  # Numerador
        total_palabras_corpus = 0    # Denominador (Sum Nd)
        epsilon = 1e-12

        if self.raw_docs:
            for d_idx, doc_words in enumerate(self.raw_docs):
                if d_idx >= theta.shape[0]: break 
                
                doc_theta = theta[d_idx] # Probabilidad de temas en este doc
                
                for word in doc_words:
                    v_idx = -1
                    if word.isdigit():
                        v_idx = int(word)
                    elif word in self.vocab_map:
                        v_idx = self.vocab_map[word]
                        
                    if v_idx != -1 and v_idx < phi.shape[1]:
                        # Esta es la parte interna: Sum(theta * phi)
                        # Probabilidad de generar esta palabra dado el modelo
                        prob_palabra = np.dot(doc_theta, phi[:, v_idx])
                        
                        # Sumamos el logaritmo (Numerador)
                        suma_log_probabilidades += np.log(prob_palabra + epsilon)
                        
                        # Contamos la palabra (Denominador)
                        total_palabras_corpus += 1
        
        if total_palabras_corpus > 0:
            resultado_final = - (suma_log_probabilidades / total_palabras_corpus)
            return resultado_final
        else:
            return None

    def generar_top_20(self):
        if not self.todos_los_topicos: return None
        self.todos_los_topicos.sort(key=lambda x: x[0])
        top_20 = self.todos_los_topicos[:20]
        
        out_path = os.path.join(self.ruta_base, "top_20_topicos.txt")
        with open(out_path, 'w', encoding="utf-8") as f:
            f.write("TOP 20 TÓPICOS (Menor Entropía es Mejor)\n")
            f.write("========================================\n")
            f.write(f"{'Modelo':<10} | {'Tópico':<10} | {'Entropía':<15}\n")
            for score, k, t_id in top_20:
                f.write(f"K={k:<8} | {t_id:<10} | {score:.5f}\n")
        return out_path


    def graficar(self, datos):
        if not datos: return
        
        keys = sorted(datos.keys())
        vals = [datos[k] for k in keys] 

        plt.figure(figsize=(10, 6))
        
        plt.plot(keys, vals, 'b-', marker='o', linewidth=2, label='Perplejidad (Log)')
        
        # Buscar el mínimo para marcarlo
        min_val = min(vals)
        min_k = keys[vals.index(min_val)]
        plt.plot(min_k, min_val, 'r*', markersize=15, label=f'Mejor K={min_k}')

        plt.title('Evaluación del Modelo (Según Fórmula de Imagen)')
        plt.xlabel('Número de Tópicos (K)')
        plt.ylabel('Log(Entropía) / Perplejidad (Menor es mejor)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.img_path)
        plt.close()

    # Actualiza ejecutar para manejar el nuevo formato de datos
    def ejecutar(self):
        lista_k = sorted([int(re.search(r'K_(\d+)', f).group(1)) 
                          for f in glob.glob(os.path.join(self.ruta_base, 'K_*')) 
                          if re.search(r'K_(\d+)', f)])
        
        if not lista_k:
            return {"success": False, "msg": "No se encontraron carpetas K_*"}
            
        if not self.cargar_datos(lista_k[0]):
            return {"success": False, "msg": "Error cargando vocabulario/corpus"}

        print(f"Evaluando K: {lista_k}")
        resultados = {}
        for k in lista_k:
            res = self.procesar_modelo(k)
            if res is not None: 
                resultados[k] = res # Guardamos el float directo
                print(f"   K={k} -> Score: {res:.4f}") # Debug en consola
            
        if not resultados:
            return {"success": False, "msg": "Fallo en cálculo de métricas."}

        txt_path = self.generar_top_20()
        self.graficar(resultados)
        
        best_k = min(resultados, key=resultados.get)
        
        return {
            "success": True, 
            "img": self.img_path, 
            "txt": txt_path,
            "best_k": best_k
        }