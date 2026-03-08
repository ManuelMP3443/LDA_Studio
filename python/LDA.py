import subprocess
import os

class LDARunner:
    def __init__(self, ruta_ejecutable: str):
        self.ruta_exec = os.path.abspath(ruta_ejecutable)

    def procesar_archivos(self, ruta_carpeta: str, k_val: int):
        """
        Ejecuta el LDA en la carpeta dada.
        Devuelve: (Exito: bool, K: int, Mensaje: str)
        """
        ruta = os.path.abspath(ruta_carpeta)
        
        # Rutas completas a los archivos
        config = os.path.join(ruta)
        corpus = os.path.join(ruta)
        vocab = os.path.join(ruta)
        resultados = os.path.join(ruta) 

        # El comando para subprocess
        comando = [
            self.ruta_exec,
            config,
            corpus,
            vocab,
            resultados
        ]
        
        try:
            # Ejecutar en el directorio correcto (cwd=ruta)
            result = subprocess.run(comando, capture_output=True, text=True, cwd=ruta)
            
            if result.returncode != 0:
                # ERROR: Devolvemos 3 cosas
                err_msg = f"Error interno: {result.stderr[:200]}" 
                return False, k_val, err_msg
            
            # ÉXITO: Devolvemos 3 cosas
            return True, k_val, "Éxito"
            
        except Exception as e:
            # EXCEPCIÓN: Devolvemos 3 cosas
            return False, k_val, str(e)