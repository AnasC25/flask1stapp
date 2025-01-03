from flask import Flask, request, render_template, send_file # type: ignore
import os
import pickle
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from datetime import datetime
from io import BytesIO

app = Flask(__name__)

# Charger le modèle DBSCAN
model_path = "dbscan_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Le fichier 'dbscan_model.pkl' est introuvable.")
else:
    with open(model_path, 'rb') as file:
        dbscan = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "Aucun fichier téléchargé", 400

    file = request.files["file"]
    if file.filename == "":
        return "Aucun fichier sélectionné", 400

    # Lire le fichier CSV
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {str(e)}", 400

    # Vérifier les colonnes nécessaires
    required_columns = ["TransactionAmount", "TransactionDate"]
    if not all(col in data.columns for col in required_columns):
        return f"Le fichier doit contenir les colonnes : {', '.join(required_columns)}", 400

    # Convertir TransactionDate en datetime
    try:
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])
    except Exception as e:
        return f"Erreur de conversion des dates : {str(e)}", 400

    # Calculer la durée
    data["Durée"] = (data["TransactionDate"] - data["TransactionDate"].min()).dt.days

    # Normaliser les données
    X = data[["TransactionAmount", "Durée"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer le modèle
    data["Cluster"] = dbscan.fit_predict(X_scaled)
    data["Statut_DBSCAN"] = data["Cluster"].apply(lambda x: "Frauduleuse" if x == -1 else "Valide")
    data["Statut"] = data["Statut_DBSCAN"]

    # Sauvegarder les résultats dans un fichier CSV en mémoire
    output = BytesIO()
    data.to_csv(output, index=False)
    output.seek(0)

    return send_file(
    output,
    mimetype="text/csv",
    as_attachment=True,
    download_name="transactions_analyzed.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)
