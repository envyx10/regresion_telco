# Telco Customer Churn Prediction

Este proyecto implementa un flujo de trabajo completo de ciencia de datos para predecir la rotación de clientes (churn) en una empresa de telecomunicaciones. El objetivo es identificar qué clientes tienen mayor probabilidad de abandonar el servicio utilizando técnicas de Machine Learning.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **`notebooks/`**: Contiene los Jupyter Notebooks con el análisis exploratorio y el desarrollo del modelo.
    - `Pruebas.ipynb`: Notebook principal con todo el flujo de trabajo comentado didácticamente.
- **`datasets/`**: Almacena los conjuntos de datos utilizados.
    - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset original de Kaggle.
- **`models/`**: Directorio para guardar los modelos entrenados serializados.
    - `churn-model.kpck`: Archivo binario que contiene el `DictVectorizer` y el modelo `LogisticRegression` entrenados.
- **`src/`**: Código fuente de la aplicación.
    - `regresion_pia/`: Paquete principal.
        - `churn_predict_service.py`: Lógica de predicción (carga modelo y transforma datos).
        - `churn_predict_app.py`: API Flask para servir el modelo.
- **`deploy/`**: Archivos relacionados con el despliegue (vacío por el momento).

## Flujo de Trabajo (Workflow)

El proceso de análisis y modelado se detalla paso a paso en `notebooks/Pruebas.ipynb`. A continuación, se resume el flujo implementado:

### 1. Preparación de Datos
- **Carga**: Se utiliza `pandas` para cargar los datos del CSV.
- **Limpieza**:
    - Conversión de `TotalCharges` a numérico (manejo de errores y nulos).
    - Normalización de nombres de columnas y valores de texto a minúsculas y sin espacios (snake_case).
    - Transformación de la variable objetivo `churn` a formato binario (0 y 1).

### 2. División de Datos
- Se divide el dataset en tres conjuntos para garantizar una evaluación honesta del modelo:
    - **Entrenamiento (Train)**: ~60%
    - **Validación (Val)**: ~20%
    - **Prueba (Test)**: ~20%

### 3. Análisis Exploratorio de Datos (EDA)
- **Tasa de Churn Global**: Cálculo de la media de abandono.
- **Análisis de Importancia**:
    - **Variables Categóricas**: Uso de la **Información Mutua (Mutual Information)** para medir la dependencia con el churn.
    - **Variables Numéricas**: Cálculo de la **Correlación de Pearson**.
- **Visualización**: Análisis de tasas de churn por grupos (ej. género, contrato).

### 4. Ingeniería de Características (Feature Engineering)
- **One-Hot Encoding**: Transformación de variables categóricas a numéricas utilizando `DictVectorizer` de Scikit-Learn. Esto convierte cada categoría en una columna binaria separada.

### 5. Entrenamiento del Modelo
- **Algoritmo**: **Regresión Logística** (`LogisticRegression`).
- **Configuración**: Solver `liblinear`, adecuado para datasets pequeños y clasificación binaria.
- **Entrenamiento**: Ajuste del modelo con los datos de entrenamiento procesados.

### 6. Evaluación
- **Predicción**: Generación de probabilidades de churn (`predict_proba`) sobre el conjunto de validación.
- **Métrica**: Cálculo de la **Exactitud (Accuracy)** comparando las predicciones (con umbral 0.5) contra los valores reales.

### 7. Serialización
- Guardado del modelo y el vectorizador en un archivo `.kpck` (pickle) para su reutilización futura sin necesidad de reentrenar.

## Instalación y Uso

Este proyecto utiliza **Poetry** para la gestión de dependencias.

### Requisitos
- Python >= 3.14
- Poetry

### Configuración del Entorno

1.  Instalar dependencias:
    ```bash
    poetry install
    ```

2.  Activar el entorno virtual:
    ```bash
    poetry shell
    ```

3.  Ejecutar Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4.  Abrir `notebooks/Pruebas.ipynb` para ver el análisis completo.

## Próximos Pasos

- Contenerizar la aplicación para su despliegue (Docker).
- Crear pruebas unitarias para el servicio de predicción.
- Mejorar la interfaz de usuario (frontend).
