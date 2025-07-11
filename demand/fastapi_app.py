from datetime import datetime, timedelta
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')
from prophet import Prophet
import pandas as pd
import numpy as np

import logging
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import psycopg2
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)


def create_holidays_df_from_db_psycopg2():
    try:
        db_host = os.getenv("DB_HOST", "takeoff.lbc")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "crms_stage_2025_06")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASS", "bitnami")

        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port
        )

        query = """
        SELECT "HolidayDate" AS "date"
        FROM "DefHoliday"
        """

        df = pd.read_sql(query, conn)
        conn.close()

        holiday_df = pd.DataFrame({
            'holiday': df['type'],
            'ds': pd.to_datetime(df['date']),
            'lower_window': 0,
            'upper_window': 0
        })

        return holiday_df

    except Exception as e:
        print(f"Erreur lors du chargement des jours fériés depuis la base : {e}")
        return pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])


# ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor()

# French holidays
holiday_df = create_holidays_df_from_db_psycopg2()


def clean_outliers(df):
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_y = df['y'].median()
    df['y'] = df['y'].apply(lambda x: median_y if x < lower_bound or x > upper_bound else x)
    logging.debug(f"Outliers cleaned. Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    return df


def week_to_date(year, week):
    first_day = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
    if first_day.weekday() > 0:
        first_day += timedelta(days=7 - first_day.weekday())
    return first_day + timedelta(weeks=week - 1)


def generate_cache_key(historical_data, future_weeks, agency_id):
    sorted_data = sorted([(entry[0], entry[1], entry[2]) for entry in historical_data])
    input_data = {
        'historicalData': sorted_data,
        'futureWeeks': future_weeks,
        'agencyId': agency_id
    }
    input_str = json.dumps(input_data, sort_keys=True)
    hash_str = hashlib.md5(input_str.encode('utf-8')).hexdigest()
    return f"forecast:agency:{agency_id}:{hash_str}"


async def compute_predictions(historical_data: List[List[int]], future_weeks: int, agency_id: int) -> List[Dict]:
    if not historical_data or len(historical_data) < 104:
        logging.warning(f"Skipping agency {agency_id}: only {len(historical_data)} weeks of data, 104 required")
        return []

    def sync_compute():
        dates = [week_to_date(entry[0], entry[1]).strftime('%Y-%m-%d') for entry in historical_data]
        counts = [entry[2] for entry in historical_data]
        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': counts})

        df = clean_outliers(df)

        model = Prophet(yearly_seasonality=10, holidays=holiday_df)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)

        future = model.make_future_dataframe(periods=future_weeks, freq='W-MON')
        forecast = model.predict(future)

        max_historical = max(counts) * 1.5
        threshold = np.percentile(df['y'], 70)
        return [
            {
                'period': row['ds'].strftime('%Y-W%W'),
                'predictedCount': min(max(0, int(row['yhat'])), int(max_historical)),
                'agencyId': agency_id,
                'highDemand': bool(row['yhat'] >= threshold)
            }
            for _, row in forecast.tail(future_weeks).iterrows()
        ]

    loop = asyncio.get_event_loop()
    predictions = await loop.run_in_executor(None, sync_compute)
    return predictions


class AgencyData(BaseModel):
    agencyId: int
    historicalData: List[List[int]]


class ForecastRequest(BaseModel):
    agenciesData: List[AgencyData]
    futureWeeks: int


@app.post('/forecast')
async def forecast(request: ForecastRequest):
    try:
        agencies_data = request.agenciesData
        future_weeks = request.futureWeeks

        if not agencies_data:
            raise HTTPException(status_code=400, detail="agenciesData is required and must not be empty")
        if future_weeks <= 0:
            raise HTTPException(status_code=400, detail="futureWeeks must be positive")

        results = await asyncio.gather(*[
            compute_predictions(agency_data.historicalData, future_weeks, agency_data.agencyId)
            for agency_data in agencies_data
        ])

        all_predictions = [pred for agency_preds in results for pred in agency_preds]

        logging.debug(f"All Predictions: {all_predictions}")
        return all_predictions

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
