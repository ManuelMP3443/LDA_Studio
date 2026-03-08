#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h> 

// --- Variables Globales (leídas desde config.txt) ---
int K, V, D, ITERACIONES;
double ALPHA, BETA;

// --- ESTRUCTURAS DE DATOS GLOBALES (PUNTEROS) ---
int **n_mk; // Matriz Documento-Tema (D x K) - n_mk[m][k]
int **n_kt; // Matriz Tema-Palabra (K x V)   - n_kt[k][t]
int *n_k;   // Vector Total-Tema (K)        - n_k[k]
int **asignaciones_tema; // Matriz de asignaciones (D x N_d)

// --- Declaraciones de Funciones ---
void cargarConfig();
void cargarCorpus(int **corpusMapeado, int *corpusLongitud);
void cargarVocabulario(char **vocabulario);
void inicializarMatrices(int *corpusLongitud);
void muestreoGibbs(int **corpusMapeado, int *corpusLongitud);
void liberarMemoria(char **vocabulario, int **corpusMapeado, int *corpusLongitud);
int muestrearNuevoTema(int d, int v, int k_actual);


void cargarConfig() {
    FILE *fp; // Puntero local
    char buffer[256];

    fp = fopen("config.txt", "r");
    if (fp == NULL) {
        perror("Error al abrir el archivo config.txt");
        exit(1); // Error fatal
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (strncmp(buffer, "K=", 2) == 0) {
            sscanf(buffer, "K=%d", &K);
        } else if (strncmp(buffer, "V=", 2) == 0) {
            sscanf(buffer, "V=%d", &V);
        } else if (strncmp(buffer, "D=", 2) == 0) {
            sscanf(buffer, "D=%d", &D);
        } else if (strncmp(buffer, "ALPHA=", 6) == 0) {
            sscanf(buffer, "ALPHA=%lf", &ALPHA);
        } else if (strncmp(buffer, "BETA=", 5) == 0) {
            sscanf(buffer, "BETA=%lf", &BETA);
        } else if (strncmp(buffer, "ITERACIONES=", 12) == 0) {
            sscanf(buffer, "ITERACIONES=%d", &ITERACIONES);
        }
    }
    fclose(fp);
    
    printf("--- Configuración Cargada ---\n");
    printf("K=%d, V=%d, D=%d, ITER=%d\n", K, V, D, ITERACIONES);
    printf("ALPHA=%.3f, BETA=%.3f\n\n", ALPHA, BETA);
}

void cargarCorpus(int **corpusMapeado, int *corpusLongitud) {
    FILE *fp;
    char buffer[10240]; // Buffer grande
    int d = 0; // Contador de documento actual

    fp = fopen("corpus.txt", "r");
    if (fp == NULL) {
        perror("Error al abrir el archivo corpus.txt"); 
        exit(1); // Error fatal
    }

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        if (d >= D) {
            printf("Advertencia: Hay más documentos en corpus.txt que en config.txt (D=%d)\n", D);
            break;
        }

        int capacidad_actual = 10;
        int *doc_actual = malloc(capacidad_actual * sizeof(int));
        if (doc_actual == NULL) {
            perror("Error en malloc inicial para doc_actual");
            exit(1);
        }
        int i = 0; // Contador de palabras

        char *token = strtok(buffer, " \n");

        while (token != NULL) {
            if (i == capacidad_actual) { 
                capacidad_actual *= 2; // Duplicar capacidad
                int *tmp = realloc(doc_actual, capacidad_actual * sizeof(int));

                if (tmp == NULL) {
                    perror("Error al reasignar memoria para doc_actual");
                    free(doc_actual);
                    exit(1);
                } else {
                    doc_actual = tmp;
                }
            }
            doc_actual[i] = atoi(token);
            i++;
            token = strtok(NULL, " \n");
        }
        corpusMapeado[d] = doc_actual;
        corpusLongitud[d] = i;
        d++;
    }
    fclose(fp);
    printf("Corpus cargado. %d documentos leídos.\n", d);
}

void cargarVocabulario(char **vocabulario) {
    FILE *fp;
    
    fp = fopen("vocab.txt", "r"); 
    if (fp == NULL) {
        perror("Error al abrir el archivo vocab.txt"); 
        exit(1);
    }

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        char *id_str = strtok(buffer, ",");
        char *palabra = strtok(NULL, "\n");
        
        if (id_str == NULL || palabra == NULL) continue;

        int id = atoi(id_str);
        if (id >= V) {
             printf("Advertencia: ID %d fuera de rango (V=%d)\n", id, V);
             continue;
        }

        vocabulario[id] = malloc(strlen(palabra) + 1);
        if (vocabulario[id] == NULL) {
            perror("Error en malloc para palabra de vocabulario");
            exit(1);
        }
        strcpy(vocabulario[id], palabra);
    }
    fclose(fp);
    printf("Vocabulario cargado.\n\n");
}


/**
 * Reserva memoria (con calloc) para las matrices globales
 * y para la matriz de asignaciones.
 */
void inicializarMatrices(int *corpusLongitud) {
    printf("Reservando memoria para matrices de conteo...\n");
    
    // n_mk (D x K) - Lleno de ceros
    n_mk = calloc(D, sizeof(int *));
    for (int d = 0; d < D; d++) {
        n_mk[d] = calloc(K, sizeof(int));
    }
    
    // n_kt (K x V) - Lleno de ceros
    n_kt = calloc(K, sizeof(int *));
    for (int k = 0; k < K; k++) {
        n_kt[k] = calloc(V, sizeof(int));
    }
    
    // n_k (K) - Lleno de ceros
    n_k = calloc(K, sizeof(int));

    // asignaciones_tema (D x N_d) - (MALLOC, no ceros)
    asignaciones_tema = malloc(D * sizeof(int *));
    for (int d = 0; d < D; d++) {
        // Reservar espacio para guardar el tema de cada palabra
        asignaciones_tema[d] = malloc(corpusLongitud[d] * sizeof(int));
    }

    if (n_mk == NULL || n_kt == NULL || n_k == NULL || asignaciones_tema == NULL) {
        perror("Error en calloc/malloc para matrices globales.");
        exit(1);
    }
    printf("Memoria reservada.\n");
}

/**
 * El motor de LDA: Fases 1 (Init) y 2 (Sampling)
 */
void muestreoGibbs(int **corpusMapeado, int *corpusLongitud) {

    // --- FASE 1: INICIALIZACIÓN ALEATORIA ---
    printf("Iniciando Fase 1: Inicialización Aleatoria...\n");
    
    for (int d = 0; d < D; d++) { // Recorrer cada documento
        int N_d = corpusLongitud[d]; // Num. de palabras en este doc
        for (int i = 0; i < N_d; i++) { // Recorrer cada palabra
            
            int v = corpusMapeado[d][i]; // El ID de la palabra (Vocabulario)
            int k = rand() % K;          // Asignar un tema aleatorio (0 a K-1)
            
            // a. Guardar la asignación
            asignaciones_tema[d][i] = k;
            
            // b. Llenar las matrices de conteo
            n_mk[d][k]++;
            n_kt[k][v]++;
            n_k[k]++;
        }
    }
    printf("Fase 1 completada. Matrices llenas con conteos aleatorios.\n\n");
    
    double *p_temas = malloc(K * sizeof(double));
    if (p_temas == NULL) {
        perror("Error en malloc para p_temas");
        exit(1);
    }

    // --- FASE 2: BUCLE DE MUESTREO DE GIBBS ---
    printf("Iniciando Fase 2: Muestreo de Gibbs...\n");
    
    for (int iter = 0; iter < ITERACIONES; iter++) {
        printf("Iteración %d / %d\n", iter + 1, ITERACIONES);
        
        for (int d = 0; d < D; d++) { // Para cada documento
            int N_d = corpusLongitud[d];
            for (int i = 0; i < N_d; i++) { // Para cada palabra
                
                int v = corpusMapeado[d][i]; // ID de la palabra
                int k_actual = asignaciones_tema[d][i];
                
                // --- a. RESTAR (El paso clave) ---
                n_mk[d][k_actual]--;
                n_kt[k_actual][v]--;
                n_k[k_actual]--;
                
                // --- b. CALCULAR PROBABILIDADES ---
                double suma_total_prob = 0.0;
                
                for (int k_nuevo = 0; k_nuevo < K; k_nuevo++) {
                    // (n_kt - 1 + beta) / (n_k - 1 + V*beta)
                    double prob1 = (n_kt[k_nuevo][v] + BETA) / (n_k[k_nuevo] + V * BETA);
                    
                    // (n_mk - 1 + alpha)
                    double prob2 = (n_mk[d][k_nuevo] + ALPHA);
                    
                    p_temas[k_nuevo] = prob1 * prob2;
                    suma_total_prob += p_temas[k_nuevo];
                }
                
                // --- c. MUESTREAR NUEVO TEMA ---
                int k_nuevo = 0;
                double r = ((double)rand() / RAND_MAX) * suma_total_prob; // Num aleatorio
                double suma_acumulada = 0.0;
                
                for (int k = 0; k < K; k++) {
                    suma_acumulada += p_temas[k];
                    if (r <= suma_acumulada) {
                        k_nuevo = k;
                        break;
                    }
                }
                
                // --- d. SUMAR (Volver a meter) ---
                asignaciones_tema[d][i] = k_nuevo;
                n_mk[d][k_nuevo]++;
                n_kt[k_nuevo][v]++;
                n_k[k_nuevo]++;
            } 
        } 
    } 
    
    printf("Fase 2 completada.\n\n");
}

void guardarResultados() {
    FILE *fp_kt, *fp_mk;
    
    fp_kt = fopen("n_kt.txt", "w");
    fp_mk = fopen("n_mk.txt", "w");
    
    if (fp_kt == NULL || fp_mk == NULL) {
        perror("Error al abrir archivos de resultados");
        exit(1);
    }
    
    // Guardar n_kt (Tema-Palabra)
    for (int k = 0; k < K; k++) {
        for (int v = 0; v < V; v++) {
            fprintf(fp_kt, "%d ", n_kt[k][v]);
        }
        fprintf(fp_kt, "\n");
    }
    
    // Guardar n_mk (Documento-Tema)
    for (int d = 0; d < D; d++) {
        for (int k = 0; k < K; k++) {
            fprintf(fp_mk, "%d ", n_mk[d][k]);
        }
        fprintf(fp_mk, "\n");
    }
    
    fclose(fp_kt);
    fclose(fp_mk);
    printf("Matrices de conteo finales guardadas en n_kt.txt y n_mk.txt\n");
}

/**
 * Libera toda la memoria que reservamos
 */
void liberarMemoria(char **vocabulario, int **corpusMapeado, int *corpusLongitud) {
    printf("\nLiberando memoria...\n");

    // a. Liberar la memoria de cada palabra en el vocabulario
    for (int i = 0; i < V; i++) {
        if (vocabulario[i] != NULL) free(vocabulario[i]);
    }
    free(vocabulario);

    // b. Liberar la memoria de cada documento (tweet)
    for (int i = 0; i < D; i++) {
        if (corpusMapeado[i] != NULL) free(corpusMapeado[i]);
    }
    free(corpusMapeado);
    
    free(corpusLongitud);

    // --- c. LIBERAR NUEVAS MATRICES ---
    for (int d = 0; d < D; d++) {
        if (n_mk[d] != NULL) free(n_mk[d]);
        if (asignaciones_tema[d] != NULL) free(asignaciones_tema[d]);
    }
    free(n_mk);
    free(asignaciones_tema);

    for (int k = 0; k < K; k++) {
        if (n_kt[k] != NULL) free(n_kt[k]);
    }
    free(n_kt);
    
    free(n_k);

    printf("Memoria liberada. Saliendo.\n");
}


// --- PUNTO DE ENTRADA PRINCIPAL ---
int main() {
    srand(time(NULL)); 
    
    // 1. Cargar config. Esto llena las variables K, V, D.
    cargarConfig();

    // 2. Reservar memoria para las estructuras de carga
    printf("Reservando memoria para %d documentos y %d palabras únicas...\n", D, V);
    int **corpusMapeado = malloc(D * sizeof(int *)); 
    int *corpusLongitud = malloc(D * sizeof(int)); 
    char **vocabulario = malloc(V * sizeof(char *));

    if (corpusMapeado == NULL || corpusLongitud == NULL || vocabulario == NULL) {
        perror("Error en el Malloc principal. ¿Demasiados datos?");
        return 1;
    }

    // 3. Llenar estas estructuras
    cargarCorpus(corpusMapeado, corpusLongitud);
    cargarVocabulario(vocabulario);

    // 4. Reservar memoria para las matrices de LDA
    inicializarMatrices(corpusLongitud);

    // 5. Ejecutar el algoritmo
    muestreoGibbs(corpusMapeado, corpusLongitud);
    
    // 6. Escribir las matrices n_mk y n_kt a archivos phi.txt y theta.txt
    printf("--- ¡ALGORITMO COMPLETADO! ---\n");
    
    printf("Ejemplo: Primera palabra del primer doc: ID=%d, Palabra='%s'\n", 
           corpusMapeado[0][0], vocabulario[corpusMapeado[0][0]]);
    printf("Longitud del primer doc: %d palabras\n", corpusLongitud[0]);
    
    guardarResultados();

    // 7. --- LIMPIEZA FINAL ---
    liberarMemoria(vocabulario, corpusMapeado, corpusLongitud);
    
    return 0; // ¡Éxito!
}