def predict_single(customer, dv, model):
    # 1. Transformación: Convierte el diccionario del cliente (datos crudos)
    # en una matriz numérica usando el DictVectorizer (dv) ya entrenado.
    x = dv.transform([customer])
    
    # 2. Predicción: Obtiene la probabilidad de que sea clase 1 (Churn).
    # model.predict_proba devuelve [[prob_no, prob_yes]], tomamos la columna 1.
    y_pred = model.predict_proba(x)[:, 1]
    
    # 3. Retorno: Devuelve una tupla con:
    # - Un booleano (True/False) si la probabilidad es >= 0.5
    # - La probabilidad numérica exacta (ej. 0.78)
    return (y_pred[0] >= 0.5, y_pred[0])