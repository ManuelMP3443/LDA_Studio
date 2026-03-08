import customtkinter as ctk
from tkinter import filedialog, messagebox
from prepros import Prepros
from LDA import LDARunner
from PIL import Image, ImageTk
from entropia import EvaluadorLDA
from resultados import ResultadosGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time

# =======================================================
# CONFIGURACIÓN VISUAL GLOBAL
# =======================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Paleta de Colores
COLOR_ACCENT = "#3B8ED0"
COLOR_SUCCESS = "#2CC985"
COLOR_WARNING = "#E5B800"
COLOR_BG_CARD = "#2B2B2B"
COLOR_CONSOLE = "#1A1A1A"
COLOR_CONSOLE_TEXT = "#00FF41"

class LDAProApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("LDA Studio")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Estado
        self.archivos_seleccionados = []
        self.modo_entrada = "csv"
        self.ruta_resultados_cargados = None
        
        # Backend
        self.pre = Prepros()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sources = os.path.join(base_dir, "sources")
        ruta_exe = os.path.join(base_dir, "sources", "LDA.exe")
        
        # Validación silenciosa al inicio (se valida mejor al ejecutar)
        if not os.path.exists(ruta_exe):
            print(f"Aviso: LDA.exe no encontrado en {ruta_exe}")

        self.LDA = LDARunner(ruta_exe)
        self.Resultados = ResultadosGenerator()

        # UI
        self.crear_panel_lateral()
        self.crear_panel_principal()

    def crear_panel_lateral(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(self.sidebar, text="LDA", font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, padx=20, pady=(30, 10))

        # --- 1. DATOS ---
        frame_data = ctk.CTkFrame(self.sidebar, fg_color=COLOR_BG_CARD)
        frame_data.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_data, text="1. Fuente de Datos", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.selector_modo = ctk.CTkSegmentedButton(frame_data, values=["CSV Único", "Múltiples Txt"], command=self.cambiar_modo_entrada)
        self.selector_modo.set("CSV Único")
        self.selector_modo.pack(padx=10, pady=5, fill="x")

        self.btn_cargar = ctk.CTkButton(frame_data, text="📂 Cargar CSV", fg_color="transparent", border_width=2, command=self.seleccionar_origen)
        self.btn_cargar.pack(padx=10, pady=10, fill="x")
        
        self.lbl_estado_archivo = ctk.CTkLabel(frame_data, text="Sin archivo", text_color="gray", font=("Arial", 11))
        self.lbl_estado_archivo.pack(pady=(0,10))

        self.entry_columna = ctk.CTkEntry(frame_data, placeholder_text="Columna (ej: content)")
        self.entry_columna.pack(padx=10, pady=(0,15), fill="x")

        # --- 2. PARÁMETROS ---
        frame_params = ctk.CTkFrame(self.sidebar, fg_color=COLOR_BG_CARD)
        frame_params.grid(row=3, column=0, padx=15, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_params, text="2. Parámetros", font=("Arial", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        ctk.CTkLabel(frame_params, text="Idioma:").grid(row=1, column=0, padx=10, sticky="w")
        self.opt_idioma = ctk.CTkOptionMenu(frame_params, values=["Español", "Inglés"], width=90)
        self.opt_idioma.grid(row=1, column=1, padx=10, pady=5)

        ctk.CTkLabel(frame_params, text="Alpha / Beta:").grid(row=2, column=0, padx=10, sticky="w")
        box_ab = ctk.CTkFrame(frame_params, fg_color="transparent")
        box_ab.grid(row=2, column=1, padx=10, pady=5)
        self.entry_alpha = ctk.CTkEntry(box_ab, width=40); self.entry_alpha.insert(0, "50")
        self.entry_alpha.pack(side="left", padx=2)
        self.entry_beta = ctk.CTkEntry(box_ab, width=40); self.entry_beta.insert(0, "0.01")
        self.entry_beta.pack(side="left", padx=2)

        ctk.CTkLabel(frame_params, text="Lista K:").grid(row=3, column=0, columnspan=2, padx=10, sticky="w")
        self.entry_k_list = ctk.CTkEntry(frame_params, placeholder_text="20, 40, 60...")
        self.entry_k_list.insert(0, "20, 40, 60, 80, 100")
        self.entry_k_list.grid(row=4, column=0, columnspan=2, padx=10, pady=(0,15), sticky="ew")

        # --- ACCIONES ---
        self.btn_run = ctk.CTkButton(self.sidebar, text="▶ INICIAR PROCESO", height=45, fg_color=COLOR_SUCCESS, hover_color="#25A56D", font=("Arial", 14, "bold"), command=self.iniciar_hilo_proceso)
        self.btn_run.grid(row=4, column=0, padx=15, pady=20, sticky="ew")

        self.btn_cargar_resultados = ctk.CTkButton(self.sidebar, text="📁 ABRIR PREVIOS", height=40, fg_color="#6B4EAF", command=self.cargar_resultados_previos)
        self.btn_cargar_resultados.grid(row=5, column=0, padx=15, pady=(0,10), sticky="ew")

        # BOTÓN MODIFICADO: Empieza deshabilitado
        self.btn_analizar = ctk.CTkButton(self.sidebar, text="📊 VER RESULTADOS", height=40, fg_color=COLOR_ACCENT, state="disabled", command=self.run_fase_2)
        self.btn_analizar.grid(row=6, column=0, padx=15, pady=(0,20), sticky="ew")

    def crear_panel_principal(self):
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=0)
        self.right_frame.grid_columnconfigure(0, weight=1) 

        self.tabs = ctk.CTkTabview(self.right_frame)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.tabs.add("📈 Gráfico")
        self.tabs.add("📋 Tabla")

        # Placeholder Gráfico
        placeholder_frame = ctk.CTkFrame(self.tabs.tab("📈 Gráfico"), fg_color="transparent")
        placeholder_frame.pack(expand=True, fill="both")
        self.lbl_plot = ctk.CTkLabel(placeholder_frame, text="📊\n\nEl gráfico de entropía aparecerá aquí\ntras ejecutar el análisis", font=("Arial", 16), text_color="gray")
        self.lbl_plot.pack(expand=True)

        self.crear_vista_tabla()

        # Consola
        self.frame_console = ctk.CTkFrame(self.right_frame, fg_color=COLOR_CONSOLE)
        self.frame_console.grid(row=1, column=0, sticky="ew", pady=(15, 0))
        
        ctk.CTkLabel(self.frame_console, text=" TERMINAL DE ESTADO", font=("Consolas", 12, "bold"), text_color="gray").pack(anchor="w", padx=10, pady=(5,0))

        self.progress_bar = ctk.CTkProgressBar(self.frame_console, height=15)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=10, pady=5)

        self.txt_log = ctk.CTkTextbox(self.frame_console, height=150, fg_color=COLOR_CONSOLE, text_color=COLOR_CONSOLE_TEXT, font=("Consolas", 12))
        self.txt_log.pack(fill="both", padx=5, pady=5)
        self.txt_log.configure(state="disabled")
        
        self.log_msg("Sistema listo. Esperando configuración...")

    def crear_vista_tabla(self):
        ctrl_frame = ctk.CTkFrame(self.tabs.tab("📋 Tabla"), fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(ctrl_frame, text="🏆 MEJORES TÓPICOS (MENOR ENTROPÍA)", font=("Arial", 14, "bold")).pack(side="left", padx=10)
        ctk.CTkButton(ctrl_frame, text="🔄 Cargar Top 20", width=140, fg_color=COLOR_ACCENT, command=self.cargar_tabla_resultados).pack(side="right")

        self.table_frame = ctk.CTkScrollableFrame(self.tabs.tab("📋 Tabla"))
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.msg_tabla = ctk.CTkLabel(self.table_frame, text="Presiona 'Cargar Top 20' tras finalizar el análisis.", text_color="gray", font=("Arial", 14))
        self.msg_tabla.pack(pady=50)

    # ... (Métodos _extraer_palabras_de_topico, cargar_tabla_resultados, _obtener_ruta_base iguales a tu versión anterior) ...
    def _extraer_palabras_de_topico(self, ruta_base, k, id_topico):
        """Busca las palabras clave de un tópico específico en su archivo K."""
        archivo_temas = os.path.join(ruta_base, f"K_{k}", "resultados_temas.txt")
        palabras_str = "No encontrado"
        
        if not os.path.exists(archivo_temas):
            return "Archivo no existe"

        try:
            with open(archivo_temas, "r", encoding="utf-8", errors="replace") as f:
                contenido = f.read()
            
            # Usamos Regex para dividir por los encabezados "=== TEMA X ==="
            import re
                    # Esta regex busca separadores. El resultado [0] es basura, [1] es el primer tema, etc.
            patrones = re.finditer(r"(=+\s*TEMA\s+(\d+)\s*=+)", contenido)

            bloques = []
            pos = []

            for match in patrones:
                pos.append((match.start(), int(match.group(2))))

            # Añadir última posición
            pos.append((len(contenido), None))

            # Crear lista de (id, texto)
            for i in range(len(pos) - 1):
                inicio, id_tema = pos[i]
                fin, _ = pos[i + 1]
                bloque_texto = contenido[inicio:fin]
                bloques.append((id_tema, bloque_texto))

            # Buscar el bloque cuyo ID real coincida
            for real_id, texto in bloques:
                if real_id == int(id_topico):
                
                    # Extraer palabras
                    clean_words = []
                    for linea in texto.split("\n"):
                        if "(" in linea and "." in linea:
                            parts = linea.split(".")
                            if len(parts) > 1:
                                palabra = parts[1].split("(")[0].strip()
                                clean_words.append(palabra)

                    # Top N palabras
                    return clean_words[:10]



        except Exception as e:
            palabras_str = f"Error lectura: {str(e)}"
            
        return palabras_str

    def cargar_tabla_resultados(self):
        """Lee top_20_topicos.txt y muestra la tabla cruzada."""
        # 1. Limpiar tabla anterior
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        ruta_base = self._obtener_ruta_base()
        if not ruta_base:
            ctk.CTkLabel(self.table_frame, text="⚠️ No hay ruta de resultados definida.", text_color="red").pack()
            return
            
        archivo_ranking = os.path.join(ruta_base, "top_20_topicos.txt")
        if not os.path.exists(archivo_ranking):
            ctk.CTkLabel(self.table_frame, text="❌ No se encontró 'top_20_topicos.txt'.\nEjecuta el análisis primero.", text_color="orange").pack()
            return

        # --- ENCABEZADOS DE LA TABLA ---
        headers = ["Rank", "Modelo (K)", "ID Tópico", "Entropía", "Palabras Clave (Contenido)"]
        anchos = [50, 80, 80, 100, 600] # Pesos relativos para grid
        
        for col, (texto, ancho) in enumerate(zip(headers, anchos)):
            lbl = ctk.CTkLabel(self.table_frame, text=texto, font=("Arial", 12, "bold"), 
                         fg_color="#404040", corner_radius=6, width=ancho)
            lbl.grid(row=0, column=col, sticky="ew", padx=2, pady=5)
            if col == 4: # La columna de palabras se expande
                self.table_frame.grid_columnconfigure(col, weight=1)

        # --- PROCESAR ARCHIVO ---
        try:
            with open(archivo_ranking, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            row_idx = 1
            for linea in lines:
                linea = linea.strip()
                if not linea or "Modelo" in linea or "===" in linea: continue # Saltar headers basura

                parts = linea.split("|")
                if len(parts) >= 3:
                    # Parsear línea: "K=20 | 14 | 4.86444"
                    k_raw = parts[0].strip().replace("K=", "")
                    id_raw = parts[1].strip()
                    entro_raw = parts[2].strip()

                    # Obtener palabras (Cruzar datos)
                    palabras = self._extraer_palabras_de_topico(ruta_base, k_raw, id_raw)

                    # --- DIBUJAR FILA ---
                    valores = [str(row_idx), f"K={k_raw}", id_raw, entro_raw, palabras]
                    colores = ["gray", "#3B8ED0", "#E5B800", "#2CC985", "white"] # Colores por columna para estilo

                    for col, val in enumerate(valores):
                        ctk.CTkLabel(self.table_frame, text=val, anchor="w" if col==4 else "center",
                                     fg_color="#2B2B2B", corner_radius=4, 
                                     text_color=colores[col] if col < 4 else "white").grid(
                            row=row_idx, column=col, sticky="ew", padx=2, pady=1)
                    
                    row_idx += 1

        except Exception as e:
            ctk.CTkLabel(self.table_frame, text=f"Error procesando ranking: {e}", text_color="red").grid(row=1, column=0)

    def _obtener_ruta_base(self):
        """Helper para obtener la ruta base de resultados."""
        if self.ruta_resultados_cargados:
            return self.ruta_resultados_cargados
        elif hasattr(self, 'datos_fase1'):
            return self.datos_fase1['base_salida_dir']
        elif self.archivos_seleccionados:
            return os.path.join(os.path.dirname(self.archivos_seleccionados[0]), "salida")
        return None

    # =======================================================
    # LOGICA VISUAL
    # =======================================================
    def log_msg(self, msg):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"> {msg}\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def cambiar_modo_entrada(self, modo):
        self.modo_entrada = "csv" if modo == "CSV Único" else "files"
        if self.modo_entrada == "csv":
            self.btn_cargar.configure(text="📂 Seleccionar CSV")
            self.entry_columna.configure(state="normal", fg_color=["#F9F9FA", "#343638"])
        else:
            self.btn_cargar.configure(text="📚 Seleccionar Archivos (TXT/PDF/DOCX)")
            self.entry_columna.configure(state="disabled", fg_color="#2B2B2B")

    def seleccionar_origen(self):
        if self.modo_entrada == "csv":
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
            if path:
                self.archivos_seleccionados = [path]
                self.lbl_estado_archivo.configure(text=f"✅ {os.path.basename(path)}", text_color=COLOR_SUCCESS)
        else:
            paths = filedialog.askopenfilenames(filetypes=[("Documentos", "*.txt *.docx *.pdf")])
            if paths:
                self.archivos_seleccionados = list(paths)
                self.lbl_estado_archivo.configure(text=f"✅ {len(paths)} archivos seleccionados", text_color=COLOR_SUCCESS)

    def cargar_resultados_previos(self):
        """Carga resultados externos y ACTIVA el botón de analizar."""
        carpeta = filedialog.askdirectory(title="Selecciona carpeta 'salida'")
        if not carpeta: return
        
        self.ruta_resultados_cargados = carpeta
        self.log_msg(f"✅ Resultados externos cargados: {os.path.basename(carpeta)}")
        self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS") # Reactivar
        
        messagebox.showinfo("Listo", "Carpeta cargada. Ahora pulsa 'Ver Resultados'.")

    # =======================================================
    # LOGICA DE EJECUCIÓN
    # =======================================================
    def iniciar_hilo_proceso(self):
        if not self.archivos_seleccionados:
            messagebox.showerror("Error", "Faltan archivos.")
            return
            
        self.btn_run.configure(state="disabled", text="⏳ PROCESANDO...")
        self.btn_analizar.configure(state="disabled") # Asegurar que esté desactivado
        self.log_msg("Iniciando hilo de procesamiento...")
        
        t = threading.Thread(target=self.worker_proceso_completo)
        t.start()

    def worker_proceso_completo(self):
        try:
            self.log_msg("--- FASE 1: Preprocesamiento ---")
            self.progress_bar.set(0.1)
            
            resultado = self.pre.run(
                ruta_entrada=self.archivos_seleccionados,
                K_TEMAS=self.entry_k_list.get(),
                Beta=float(self.entry_beta.get()),
                Idioma="english" if self.opt_idioma.get() == "Inglés" else "spanish",
                flag_csv=(self.modo_entrada == "csv"),
                columna_csv=self.entry_columna.get(),
                alfa=float(self.entry_alpha.get())
            )

            if not resultado["success"]:
                self.log_msg(f"ERROR FASE 1: {resultado['error']}")
                self.btn_run.configure(state="normal", text="▶ INICIAR PROCESO")
                return

            self.log_msg(f"Preprocesamiento OK. Vocabulario: {resultado['V']}")
            self.datos_fase1 = resultado
            self.progress_bar.set(0.2)

            configs = resultado["resultados_por_k"]
            total_k = len(configs)
            max_workers = os.cpu_count() or 4
            
            self.log_msg(f"--- FASE LDA: Iniciando {total_k} modelos ({max_workers} hilos) ---")
            
            futures = []
            exitos = 0
            completados = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for k, info in configs.items():
                    futures.append(executor.submit(self.LDA.procesar_archivos, info["path"], k))

                for future in as_completed(futures):
                    success, k_val, msg = future.result()
                    completados += 1
                    if success:
                        self.log_msg(f"   ✅ K={k_val} Terminado.")
                        exitos += 1
                    else:
                        self.log_msg(f"   ❌ K={k_val} FALLÓ: {msg}")
                    self.progress_bar.set(0.2 + (0.8 * (completados / total_k)))

            self.log_msg(f"Proceso finalizado. Éxitos: {exitos}/{total_k}")
            
            if exitos > 0:
                # ACTIVAR BOTÓN DE RESULTADOS AL FINALIZAR
                self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS")
                messagebox.showinfo("Completado", "El procesamiento ha terminado.\nPulsa 'Ver Resultados'.")
            
        except Exception as e:
            self.log_msg(f"ERROR CRÍTICO: {e}")
            messagebox.showerror("Error", str(e))
        
        finally:
            self.btn_run.configure(state="normal", text="▶ INICIAR PROCESO")

    def run_fase_2(self):
        """Fase de Análisis (Generación de gráficos)."""
        self.btn_analizar.configure(state="disabled", text="⏳ CARGANDO...") # Feedback Visual
        self.log_msg("--- FASE 2: Análisis y Gráficas ---")
        
        ruta_base = self._obtener_ruta_base()
        if not ruta_base:
            self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS")
            messagebox.showerror("Error", "No hay datos.")
            return

        self.log_msg(f"Analizando en: {ruta_base}")
        threading.Thread(target=self._worker_fase_2, args=(ruta_base,)).start()

    def _worker_fase_2(self, ruta_base):
        try:
            generador = ResultadosGenerator()
            res_gen = generador.procesar_carpeta_salida(ruta_base)
            if res_gen["success"]:
                self.log_msg(f"Reportes generados: {res_gen['procesados']}")

            evaluador = EvaluadorLDA(ruta_base)
            res = evaluador.ejecutar()

            if res["success"]:
                self.log_msg(f"¡Análisis listo! Mejor K estimado: {res['best_k']}")
                # Usamos after para manipular UI desde el hilo principal
                self.after(0, lambda: self._mostrar_resultados_ui(res))
            else:
                self.log_msg(f"Error Análisis: {res['msg']}")
                self.after(0, lambda: self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS"))
                
        except Exception as e:
            self.log_msg(f"Error Crítico F2: {e}")
            self.after(0, lambda: self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS"))

    def _mostrar_resultados_ui(self, res):
        self._mostrar_grafico(res["img"])
        self.btn_analizar.configure(state="normal", text="📊 VER RESULTADOS") # Restaurar botón
        
        if 'best_k' in res:
            self.cargar_tabla_resultados()
        
        messagebox.showinfo("Finalizado", f"Análisis completado.\nMejor modelo sugerido: K={res['best_k']}")
        self.tabs.set("📈 Gráfico")

    def _mostrar_grafico(self, img_path):
        """Muestra el gráfico liberando el archivo."""
        if not os.path.exists(img_path): return
        try:
            with Image.open(img_path) as img_file:
                img_file.load()
                pil_img = img_file.copy()
            
            self.tabs.update_idletasks()
            cw = self.tabs.tab("📈 Gráfico").winfo_width() or 800
            ch = self.tabs.tab("📈 Gráfico").winfo_height() or 600
            
            ratio = min((cw-40)/pil_img.width, (ch-40)/pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=new_size)
            self.lbl_plot.configure(image=ctk_img, text="")
            
        except Exception as e:
            print(f"Error img: {e}")

if __name__ == "__main__":
    app = LDAProApp()
    app.mainloop()