# Explicación Detallada de `Pruebas.ipynb`

Este documento explica paso a paso el código y la lógica implementada en el notebook `notebooks/Pruebas.ipynb` para la predicción de churn.

## 1. Preparación de los Datos

### Carga y Limpieza Inicial
El proceso comienza importando `pandas` y cargando el dataset.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('./datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

**Limpieza de `TotalCharges`**:
La columna `TotalCharges` se lee inicialmente como texto porque contiene espacios en blanco para valores nulos. Se fuerza la conversión a numérico (`errors='coerce'`), convirtiendo esos errores en `NaN` (Not a Number), que luego se rellenan con 0.

```python
# Convertir a numérico, los errores se vuelven NaN
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
# Rellenar NaN con 0
df.TotalCharges = df.TotalCharges.fillna(0)
```

**Normalización de Texto**:
Se estandarizan los nombres de las columnas y el contenido de las variables categóricas a minúsculas y reemplazando espacios por guiones bajos (`_`). Esto evita errores por inconsistencias (ej. "Yes" vs "yes").

```python
# Función lambda para normalizar texto
replacer = lambda str: str.lower().str.replace(' ', '_')

# Normalizar nombres de columnas
df.columns = replacer(df.columns.str)

# Normalizar valores de columnas de texto
for col in list(df.dtypes[df.dtypes == 'object'].index):
    df[col] = replacer(df[col].str)
```

### Transformación de la Variable Objetivo
La variable `churn` (abandono) tiene valores "yes" y "no". Se convierte a números enteros (1 y 0) para que el modelo pueda procesarla.

```python
df.churn = (df.churn == 'yes').astype(int)
```

## 2. División del Dataset (Splitting)

Para evaluar correctamente el modelo, los datos se dividen en tres partes usando `train_test_split` de Scikit-Learn:

```python
from sklearn.model_selection import train_test_split

# 1. Separar 20% para Test (df_test) y 80% para Entrenamiento Completo (df_train_full)
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

# 2. Del 80% restante, separar un 33% (aprox 20% del total) para Validación (df_val)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=1)

# 3. Extraer la variable objetivo (y) y eliminarla de los datos de entrenamiento (X)
y_train = df_train.churn.values
y_val = df_val.churn.values
del df_train['churn']
del df_val['churn']
```

## 3. Análisis de Importancia de Características (Feature Importance)

### Variables Categóricas - Información Mutua
La **Información Mutua (Mutual Information)** mide el grado de dependencia entre una variable categórica y el churn.

```python
from sklearn.metrics import mutual_info_score

def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
```
*Interpretación*: Variables con mayor MI (como `contract`, `onlinesecurity`) son más predictivas.

### Variables Numéricas - Correlación
La **Correlación de Pearson** mide la relación lineal.

```python
df_train_full[numerical].corrwith(df_train_full.churn)
```
*   `tenure` (antigüedad): Correlación negativa. A mayor antigüedad, menor probabilidad de churn.
*   `monthlycharges`: Correlación positiva. A mayor pago mensual, mayor probabilidad de churn.

## 4. Ingeniería de Características (Feature Engineering)

Los modelos de Machine Learning necesitan números, no texto. Usamos **One-Hot Encoding** con `DictVectorizer`.

1.  **Convertir a Diccionarios**: Transformamos el dataframe en una lista de diccionarios.
2.  **Vectorizar**: `DictVectorizer` convierte cada par "clave: valor" en una columna binaria.

```python
from sklearn.feature_extraction import DictVectorizer

# Convertir datos de entrenamiento a diccionarios
train_dict = df_train[categorical + numerical].to_dict(orient='records')

# Inicializar y ajustar el vectorizador
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

# Transformar los diccionarios a matriz numérica
X_train = dv.transform(train_dict)
```

## 5. Entrenamiento del Modelo

Se utiliza **Regresión Logística**.

```python
from sklearn.linear_model import LogisticRegression

# Inicializar el modelo (solver='liblinear' es bueno para datasets pequeños)
model = LogisticRegression(solver='liblinear', random_state=1)

# Entrenar el modelo con los datos procesados
model.fit(X_train, y_train)
```

## 6. Evaluación del Modelo

Probamos el modelo con el conjunto de validación (`X_val`).

```python
# 1. Preparar datos de validación (usando el MISMO vectorizador ajustado antes)
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

# 2. Predecir probabilidades (columna 1 es la probabilidad de churn)
y_pred = model.predict_proba(X_val)[:, 1]

# 3. Convertir a decisión binaria (Umbral 0.5)
churn_decision = y_pred >= 0.5

# 4. Calcular exactitud (Accuracy)
accuracy = (y_val == churn_decision).mean()
print(f"Accuracy: {accuracy}")
```

## 7. Serialización (Guardado)

Guardamos el modelo y el vectorizador para usarlos en producción.

```python
import pickle

with open('models/churn-model.kpck', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
```
Esto crea un archivo binario que contiene toda la "inteligencia" del modelo.
