from flask import Flask, request, jsonify
from flask_restful import Resource, Api

import joblib
import pandas as pd

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

api = Api(app)

# Chargement du modèle d'apprentissage automatique
model = joblib.load(open('model.pkl', 'rb'))

# Point de terminaison racine
@app.route('/')
def home():
    return 'Processus de recrutement API'

# Point de terminaison de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données de la requête
    processus = request.json
    valeurs_processus = [list(d.values()) for d in processus]
    print("valeurs_processus : ", valeurs_processus)
    # Conversion des données en DataFrame
    query_df = pd.DataFrame(processus)

    print("Query : ", query_df)
    # Prédiction du classement des joueurs
    prediction = model.predict(valeurs_processus)

    # Renvoi des prédictions sous forme de JSON
    # 
    return jsonify(list(prediction))
if __name__ == '__main__':
    app.run(debug=True)
