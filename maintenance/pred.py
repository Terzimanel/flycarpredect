import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
# Configuration de l'application Flask
app = Flask(__name__)




def extract_data(agency_ids=None):
    try:
        # Get database connection parameters from environment variables
        db_host = os.getenv("DB_HOST", "host.docker.internal")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "Flycar_db")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASS", "manel")

        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port
        )
        # Calculer la date d'il y a 3 ans
        three_years_ago = datetime.now().date() - timedelta(days=3*365)
        
        # Requête pour les données de maintenance et véhicules avec calcul direct de RentalCount
        query_maintenance = """
        SELECT m."MaintenanceId", m."MaintenanceDate", m."RevisionKM", m."NextRevision", 
               v."VehicleId", v."Mileage", v."StartOfServiceDate", v."ModelId", v."RegistrationNumber", v."AgencyId",
               (
                   SELECT COUNT(*)
                   FROM "Contract" c
                   JOIN "VehicleContractMap" vcm ON c."ContractId" = vcm."ContractId"
                   WHERE vcm."VehicleId" = v."VehicleId"
                   AND c."PickupDate" < m."MaintenanceDate"
                   AND c."PickupDate" >= %s
               ) AS "RentalCount"
        FROM "DefMaintenance" m
        JOIN "DefVehicle" v ON m."VehicleId" = v."VehicleId"
        WHERE m."MaintenanceDate" >= %s
        """
        params = (three_years_ago, three_years_ago)
        
        if agency_ids:
            query_maintenance += f' AND v."AgencyId" IN %s'
            params += (tuple(agency_ids),)
            
        df_maintenance = pd.read_sql(query_maintenance, conn, params=params)
        df_maintenance['AgeInDays'] = df_maintenance['StartOfServiceDate'].apply(
            lambda x: (datetime.now().date() - x).days if pd.notnull(x) else 0
        )
        
        df_maintenance.to_csv('maintenance_data.csv', index=False)
        conn.close()
        print(f"Data extracted and saved to maintenance_data.csv for agencies {agency_ids if agency_ids else 'all'}")
        print(f"Data limited to the last 3 years (since {three_years_ago})")
        return df_maintenance
    except Exception as e:
        print(f"Error during data extraction: {e}")
        return None






# Fonction pour gérer les valeurs aberrantes avec la méthode IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_rows = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed {initial_rows - len(df)} outliers from {column}")
    return df

# Fonction pour préparer les données pour l'entraînement
def prepare_data(df):
    try:
        df['Mileage'] = df['Mileage'].fillna(0)
        df['RevisionKM'] = df['RevisionKM'].fillna(0)
        df['NextRevision'] = df['NextRevision'].fillna(0)
        df['AgeInDays'] = df['AgeInDays'].fillna(0)
        df['RentalCount'] = df['RentalCount'].fillna(0)  # Gestion des valeurs manquantes pour RentalCount

        # Créer la variable cible et limiter les valeurs extrêmes
        df['Target'] = df['NextRevision'] - df['Mileage']
        df['Target'] = df['Target'].clip(lower=0, upper=100000)

        # Gestion des valeurs aberrantes pour les colonnes clés
        for col in ['Mileage', 'AgeInDays', 'RevisionKM', 'NextRevision', 'Target']:
            df = handle_outliers(df, col)

        # Visualisation de la distribution de Target
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Target'], bins=30, kde=True)
        plt.title('Distribution of Target (Remaining KM to Next Maintenance)')
        plt.xlabel('Remaining KM')
        plt.ylabel('Frequency')
        plt.savefig('target_distribution.png')
        plt.close()
        print("Distribution plot of Target saved as target_distribution.png")

        print("\nBasic statistics of Target:")
        print(df['Target'].describe())

        features = ['Mileage', 'AgeInDays', 'RevisionKM']  # RentalCount n'est pas inclus comme feature pour l'instant
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Data prepared for training")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return None, None, None, None, None, None


# Fonction pour entraîner et choisir le meilleur modèle avec optimisation des hyperparamètres



def train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    try:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }
        best_model = None
        best_score = float('inf')
        best_model_name = ""

        for name, model in models.items():
            print(f"Training {name}...")
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters for Random Forest: {grid_search.best_params_}")
            elif name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.9]
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters for XGBoost: {grid_search.best_params_}")
            elif name == 'LightGBM':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.9]
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters for LightGBM: {grid_search.best_params_}")
            elif name == 'CatBoost':
                param_grid = {
                    'iterations': [50, 100, 200],
                    'depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters for CatBoost: {grid_search.best_params_}")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"{name} - RMSE: {rmse}, R2: {r2}")
            
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = name

        # Ajout d'un modèle de stacking
        print("Training Stacking Regressor...")
        estimators = [
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1)),
            ('xgb', XGBRegressor(random_state=42, n_jobs=-1)),
            ('lgbm', LGBMRegressor(random_state=42, n_jobs=-1))
        ]
        stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1)
        stacking_model.fit(X_train_scaled, y_train)
        stacking_pred = stacking_model.predict(X_test_scaled)
        stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
        stacking_r2 = r2_score(y_test, stacking_pred)
        print(f"Stacking Regressor - RMSE: {stacking_rmse}, R2: {stacking_r2}")
        
        if stacking_rmse < best_score:
            best_score = stacking_rmse
            best_model = stacking_model
            best_model_name = "Stacking Regressor"

        joblib.dump(best_model, 'maintenance_predictor.pkl')
        print(f"Best model: {best_model_name} with RMSE: {best_score}")
        return best_model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None


# Fonction pour évaluer et valider le modèle
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    try:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"Cross-validation RMSE: {cv_rmse}")

        y_pred = model.predict(X_test_scaled)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        final_r2 = r2_score(y_test, y_pred)
        print(f"Final Test RMSE: {final_rmse}")
        print(f"Final Test R2: {final_r2}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")

# Endpoint Flask pour les prédictions par véhicule individuel
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_features = ['Mileage', 'AgeInDays', 'RevisionKM']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400
        
        features = [data['Mileage'], data['AgeInDays'], data['RevisionKM']]
        scaler = joblib.load('scaler.pkl')
        features_scaled = scaler.transform([features])
        
        model = joblib.load('maintenance_predictor.pkl')
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({'prediction': float(prediction)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Nouveau endpoint Flask pour prédire les maintenances par agence
# Nouveau endpoint Flask pour prédire les maintenances par agence
@app.route('/predict_agency', methods=['POST'])
def predict_agency():
    try:
        data = request.get_json()
        if 'agency_ids' not in data or not isinstance(data['agency_ids'], list):
            return jsonify({'error': 'Missing or invalid agency_ids parameter, must be a list'}), 400
        
        agency_ids = data['agency_ids']
        if not agency_ids:
            return jsonify({'error': 'agency_ids list cannot be empty'}), 400
        
        df = extract_data(agency_ids)
        if df is None or df.empty:
            return jsonify({'error': f'No data found for agencies {agency_ids}'}), 404

        # Préparer les données pour la prédiction
        df['Mileage'] = df['Mileage'].fillna(0)
        df['AgeInDays'] = df['AgeInDays'].fillna(0)
        df['RevisionKM'] = df['RevisionKM'].fillna(0)
        df['RentalCount'] = df['RentalCount'].fillna(0)

        features = ['Mileage', 'AgeInDays', 'RevisionKM']
        X = df[features]
        
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)
        
        model = joblib.load('maintenance_predictor.pkl')
        predictions = model.predict(X_scaled)
        
        # Estimation des jours restants (hypothèse : 50 km par jour en moyenne)
        km_per_day = 50
        results_by_agency = {}
        for i, pred in enumerate(predictions):
            agency_id = int(df['AgencyId'].iloc[i])
            if agency_id not in results_by_agency:
                results_by_agency[agency_id] = []
            
            km_remaining = float(pred)
            days_remaining = max(0, int(km_remaining / km_per_day)) if km_remaining > 0 else 0
            results_by_agency[agency_id].append({
                'VehicleId': int(df['VehicleId'].iloc[i]),
                'ModelId': str(df['ModelId'].iloc[i]) if pd.notnull(df['ModelId'].iloc[i]) else 'N/A',
                'LicensePlate': str(df['RegistrationNumber'].iloc[i]) if pd.notnull(df['RegistrationNumber'].iloc[i]) else 'N/A',
                'KmRemaining': km_remaining,
                'DaysRemaining': days_remaining,
                'RentalCount': int(df['RentalCount'].iloc[i])  # Ajout du nombre de locations avant maintenance
            })
        
        return jsonify({'results_by_agency': results_by_agency}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500





# Fonction principale pour gérer les modes
def main(mode):
    if mode == "train":
        df = extract_data()
        if df is None:
            return

        X_train_scaled, X_test_scaled, y_train, y_test, scaler, cleaned_df = prepare_data(df)
        if X_train_scaled is None:
            return

        joblib.dump(scaler, 'scaler.pkl')
        best_model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        if best_model is None:
            return

        evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
    elif mode == "server":
        # Check if model and scaler files exist; train if they don't
        if not os.path.exists('maintenance_predictor.pkl') or not os.path.exists('scaler.pkl'):
            print("Model or scaler not found. Training the model...")
            df = extract_data()
            if df is None:
                return

            X_train_scaled, X_test_scaled, y_train, y_test, scaler, cleaned_df = prepare_data(df)
            if X_train_scaled is None:
                return

            joblib.dump(scaler, 'scaler.pkl')
            best_model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
            if best_model is None:
                return

            evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)

        print("Starting Flask API server...")
        app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maintenance Predictor Script")
    parser.add_argument('--mode', type=str, choices=['train', 'server'], default='train',
                        help="Mode to run the script: 'train' for training the model, 'server' for running the API")
    args = parser.parse_args()
    main(args.mode)
