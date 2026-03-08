import numpy as np
import os
import glob
import re

class ResultadosGenerator:
    def __init__(self):
        # Configuración interna
        self.N_PALABRAS_TOP = 20
        self.ARCH_CONFIG = "config.txt"
        self.ARCH_VOCAB = "vocab.txt"
        self.ARCH_N_KT = "n_kt.txt"
        self.ARCH_RESULTADOS = "resultados_temas.txt"

    def _cargar_config(self, ruta_carpeta):
        """Lee el config.txt de una carpeta específica."""
        params = {}
        path = os.path.join(ruta_carpeta, self.ARCH_CONFIG)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for linea in f:
                    if '=' in linea:
                        clave, valor = linea.strip().split('=')
                        params[clave] = float(valor) if '.' in valor else int(valor)
            return params
        except Exception:
            return None

    def _cargar_vocab(self, ruta_carpeta):
        """Lee vocab.txt."""
        vocab = []
        path = os.path.join(ruta_carpeta, self.ARCH_VOCAB)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for linea in f:
                    if ',' in linea:
                        vocab.append(linea.strip().split(',', 1)[1])
            return vocab
        except Exception:
            return None

    def _cargar_matriz(self, ruta_carpeta, nombre_archivo):
        """Carga matriz usando numpy."""
        path = os.path.join(ruta_carpeta, nombre_archivo)
        try:
            matriz = np.loadtxt(path, dtype=np.int32)
            if len(matriz.shape) == 1: matriz = matriz.reshape(1, -1)
            return matriz
        except Exception:
            return None

    def _calcular_phi(self, n_kt, beta, V):
        """Fórmula de Bayes para Phi."""
        n_k = n_kt.sum(axis=1)
        # (n_kt + beta) / (n_k + V * beta)
        return (n_kt + beta) / (n_k[:, np.newaxis] + V * beta)

    def _generar_txt_legible(self, ruta_carpeta, phi, vocab, k_val):
        """Escribe el archivo resultados_temas.txt."""
        path_salida = os.path.join(ruta_carpeta, self.ARCH_RESULTADOS)
        K = phi.shape[0]
        try:
            with open(path_salida, 'w', encoding='utf-8') as f:
                f.write(f"--- ANÁLISIS DE TEMAS LDA (K={k_val}) ---\n")
                for k in range(K):
                    f.write(f"\n==================== TEMA {k} ====================\n")
                    indices_top = np.argsort(phi[k])[::-1][:self.N_PALABRAS_TOP]
                    for i, idx in enumerate(indices_top):
                        if idx < len(vocab):
                            f.write(f"  {i+1}. {vocab[idx]:<20} (Prob: {phi[k][idx]:.6f})\n")
            return True
        except Exception:
            return False

    # --- MÉTODO PÚBLICO PARA LLAMAR DESDE LA INTERFAZ ---
    def procesar_carpeta_salida(self, directorio_base):
        """
        Recorre la carpeta 'salida' buscando subcarpetas K_10, K_20...
        y genera los reportes en cada una.
        """
        if not os.path.exists(directorio_base):
            return {"success": False, "msg": f"No existe el directorio: {directorio_base}"}

        # Buscar carpetas K_*
        patron = os.path.join(directorio_base, "**", "K_*")
        carpetas_k = glob.glob(patron, recursive=True)
        carpetas_k = [c for c in carpetas_k if os.path.isdir(c)]

        if not carpetas_k:
            return {"success": False, "msg": "No se encontraron carpetas de resultados (K_*)."}

        procesados = 0
        errores = 0
        detalles = []

        for carpeta in carpetas_k:
            nombre = os.path.basename(carpeta)
            
            # 1. Cargar datos
            config = self._cargar_config(carpeta)
            vocab = self._cargar_vocab(carpeta)
            n_kt = self._cargar_matriz(carpeta, self.ARCH_N_KT)

            if not config or not vocab or n_kt is None:
                errores += 1
                detalles.append(f"{nombre}: Faltan archivos (¿Falló LDA?)")
                continue

            # 2. Calcular y Guardar
            try:
                phi = self._calcular_phi(n_kt, config['BETA'], len(vocab))
                if self._generar_txt_legible(carpeta, phi, vocab, config['K']):
                    procesados += 1
                else:
                    errores += 1
            except Exception as e:
                errores += 1
                detalles.append(f"{nombre}: Error cálculo ({str(e)})")

        return {
            "success": True,
            "procesados": procesados,
            "errores": errores,
            "carpetas_encontradas": len(carpetas_k),
            "msg": f"Reportes generados: {procesados} | Errores: {errores}"
        }