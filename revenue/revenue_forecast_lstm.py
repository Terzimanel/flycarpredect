import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RevenueForecaster:
    def __init__(self, contract_repository, look_back: int = 6, future_months: int = 12):
        """
        Initialize the RevenueForecaster.
        """
        self.contract_repository = contract_repository
        self.look_back = look_back
        self.future_months = future_months
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def fetch_historical_data(self, agency_ids: List[int], vehicle_type_ids: List[int], 
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical monthly revenue data from the repository.
        """
        logger.info(f"Fetching historical revenue data for agencies {agency_ids}")
        try:
            results = self.contract_repository.getMonthlyRevenueByPeriod(
                agency_ids, vehicle_type_ids, start_date, end_date
            )
            logger.debug("Raw query results: %s", results)
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}", exc_info=True)
            raise

        data = []
        for result in results:
            try:
                agency_id = int(result[0])
                year = int(result[1])
                month = int(result[2])
                revenue = float(result[3]) if result[3] is not None else 0.0
                date = datetime(year, month, 1)
                data.append([agency_id, date, revenue])
            except (ValueError, IndexError) as e:
                logger.error(f"Error processing result {result}: {str(e)}", exc_info=True)
                continue

        df = pd.DataFrame(data, columns=['agency_id', 'date', 'revenue'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['agency_id', 'date'])
        logger.info(f"Fetched {len(df)} records")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prepare data for LSTM by creating sequences for each agency.
        """
        if df.empty:
            logger.warning("No historical data to prepare")
            return np.array([]), np.array([]), {}
            
        agency_groups = {}
        X, y = [], []
        current_index = 0
        
        for agency_id in df['agency_id'].unique():
            agency_id = int(agency_id)
            agency_data = df[df['agency_id'] == agency_id][['revenue']].values
            logger.debug(f"Agency {agency_id} has {len(agency_data)} records")
            if len(agency_data) < self.look_back + 1:
                logger.warning(f"Insufficient data for agency {agency_id}: {len(agency_data)} records")
                continue
                
            scaled_data = self.scaler.fit_transform(agency_data)
            
            for i in range(len(scaled_data) - self.look_back):
                X.append(scaled_data[i:(i + self.look_back), 0])
                y.append(scaled_data[i + self.look_back, 0])
                
            agency_groups[agency_id] = list(range(current_index, len(X)))
            current_index = len(X)
        
        X = np.array(X)
        y = np.array(y)
        if X.size > 0:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        logger.info(f"Prepared {len(X)} sequences across {len(agency_groups)} agencies")
        return X, y, agency_groups

    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        try:
            self.model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            logger.info("LSTM model compiled")
        except Exception as e:
            logger.error(f"Error building model: {str(e)}", exc_info=True)
            raise

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """
        Train the LSTM model.
        """
        if X.size == 0 or y.size == 0:
            logger.error("Cannot train model: No training data available")
            raise ValueError("No training data available")
        logger.info(f"Starting model training with {epochs} epochs")
        try:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            raise

    def predict_future(self, df: pd.DataFrame, agency_groups: Dict) -> Dict[int, List[Dict]]:
        """
        Predict future revenues for each agency.
        """
        predictions = {}

        for agency_id in agency_groups.keys():
            agency_id = int(agency_id)
            predictions[agency_id] = []

            agency_data = df[df['agency_id'] == agency_id][['revenue']].values
            if len(agency_data) < self.look_back:
                logger.warning(f"Skipping predictions for agency {agency_id} due to insufficient data")
                continue

            current_sequence = agency_data[-self.look_back:]
            current_sequence = self.scaler.transform(current_sequence)
            current_sequence = np.reshape(current_sequence, (1, self.look_back, 1))

            last_date = df[df['agency_id'] == agency_id]['date'].max()

            for _ in range(self.future_months):
                try:
                    predicted = self.model.predict(current_sequence, verbose=0)
                    predicted_value = self.scaler.inverse_transform(predicted)[0, 0]

                    next_date = last_date + relativedelta(months=1)
                    predictions[agency_id].append({
                        'date': next_date.strftime('%Y-%m-%d'),
                        'predicted_revenue': float(predicted_value)
                    })

                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, 0] = predicted[0, 0]
                    last_date = next_date

                except Exception as e:
                    logger.error(f"Error predicting for agency {agency_id}: {str(e)}", exc_info=True)
                    break

        logger.info(f"Generated predictions for {len(predictions)} agencies")
        return predictions

    def forecast_revenue(self, agency_ids: List[int], vehicle_type_ids: List[int], 
                         start_date: datetime, end_date: datetime) -> Dict:
        """
        Main method to forecast future revenues.
        """
        try:
            df = self.fetch_historical_data(agency_ids, vehicle_type_ids, start_date, end_date)
            
            if df.empty:
                logger.error("No historical data available")
                return {"error": "No historical data available", "predictions": {}, "model_info": {}}
                
            X, y, agency_groups = self.prepare_data(df)
            
            if len(X) == 0 or not agency_groups:
                logger.error("Insufficient data for training")
                return {
                    "error": "Insufficient data for training",
                    "predictions": {},
                    "model_info": {
                        "look_back": self.look_back,
                        "future_months": self.future_months,
                        "training_samples": 0
                    }
                }
                
            self.build_model()
            self.train_model(X, y)
            predictions = self.predict_future(df, agency_groups)
            
            return {
                "predictions": predictions,
                "model_info": {
                    "look_back": self.look_back,
                    "future_months": self.future_months,
                    "training_samples": len(X)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in forecast_revenue: {str(e)}", exc_info=True)
            return {
                "error": f"Server error: {str(e)}",
                "predictions": {},
                "model_info": {}
            }
