import pickle

from flask import Flask, jsonify, request

from churn_predict_service import predict_single

app = Flask('churn-predict')

# Cargar el modelo y el vectorizador al iniciar la aplicación
# Se utiliza el archivo generado previamente en el notebook
with open('models/churn-model.kpck', 'rb') as f:
    dv, model = pickle.load(f)


# Definir el endpoint '/predict' que acepta peticiones POST
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Recibir los datos del cliente en formato JSON
    customer = request.get_json()
    
    # 2. Usar el servicio para obtener la predicción (lógica de negocio)
    churn, prediction = predict_single(customer, dv, model)

    # 3. Preparar la respuesta
    result = {
        'churn': bool(churn),            # Decisión binaria (True/False)
        'churn_probability': float(prediction), # Probabilidad exacta
    }

    # 4. Devolver la respuesta en formato JSON
    return jsonify(result)


if __name__ == '__main__':
    # Ejecutar el servidor en modo debug en el puerto 8000
    app.run(debug=True, port=8000)