# Latent Dirichlet Allocation (LDA) Studio

Una implementación del modelo estocástico Latent Dirichlet Allocation (LDA) para el descubrimiento de tópicos (Topic Modeling) en corpus de texto. 

Este proyecto combina el alto rendimiento de **C** para el cálculo probabilístico riguroso (Muestreo de Gibbs) con un robusto pipeline de Procesamiento de Lenguaje Natural (NLP) y una interfaz gráfica desarrollada en **Python**.

## Características Principales

* **Backend (C):** Implementación manual del Muestreo de Gibbs. Gestión dinámica de memoria mediante punteros para procesar eficientemente matrices de documentos y palabras sin depender de librerías externas para la matemática central.
* **Pipeline NLP (Python):** Preprocesamiento de texto avanzado utilizando `spaCy` y `NLTK`. Incluye limpieza, lematización, eliminación de stopwords y vectorización del corpus.
* **Evaluación Automática:** Cálculo de la métrica de Entropía para evaluar modelos y sugerir matemáticamente el número óptimo de tópicos (K) en un conjunto de datos.
* **Interfaz Gráfica (GUI):** Entorno visual interactivo construido con `CustomTkinter` que permite cargar documentos, ajustar hiperparámetros (Alpha, Beta, iteraciones) y visualizar los gráficos de convergencia y las palabras clave de forma intuitiva.

## Arquitectura y Tecnologías

* **Motor Matemático:** `C` (Compilador GCC)
* **Lógica y NLP:** `Python 3.x`
* **Librerías Clave:** * `numpy`, `matplotlib`, `polars` (Manejo de datos y visualización)
  * `nltk`, `spacy` (Modelo `es_core_news_sm` para procesamiento en español)
  * `customtkinter`, `Pillow` (Interfaz gráfica)

## Estructura del Repositorio

* `/sources/`: Código fuente del motor estadístico (`LDA.c`).
* `/python/`: Scripts de integración y lógica superior (`prepros.py`, `LDA.py`, `entropia.py`, `resultados.py` y `main.py` para la ejecución de la interfaz).
* `/ejemplos/`: Resultados de muestra en texto plano.
* `Manual de Usuario LDA.pdf`: Documentación sobre el flujo, configuración de parámetros y uso de la herramienta.
