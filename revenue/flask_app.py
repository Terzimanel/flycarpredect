from flask import Flask, request, jsonify
from revenue_forecast_lstm import RevenueForecaster
import logging
from sqlalchemy import create_engine, text
from datetime import datetime
from dateutil.parser import parse
from typing import List
import traceback
import numpy as np
import os

# Limit BLAS threads (for TensorFlow, NumPy, etc.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database configuration from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "manel")
DB_HOST = os.getenv("DB_HOST", "takeoff.lbc")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crms_stage_2025_06")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class ContractRepository:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def getMonthlyRevenueByPeriod(
        self,
        agency_ids: List[int],
        vehicle_type_ids: List[int],
        start_date: datetime,
        end_date: datetime
    ):
        # Build dynamic WHERE clause
        where_clauses = [
            'c."AgencyId" = ANY(:agency_ids)',
            'c."PickupDate" >= :start_date',
            'c."ReturnDate" <= :end_date',
            'c."ContractStatus" NOT IN (\'lblCancelled\')'
        ]

        params = {
            'agency_ids': agency_ids if agency_ids else [-1],
            'start_date': start_date,
            'end_date': end_date
        }

        if vehicle_type_ids:
            where_clauses.append(
                '(dvc."VehicleCategoryId" = ANY(:vehicle_type_ids) OR dvc."VehicleCategoryId" IS NULL)'
            )
            params['vehicle_type_ids'] = vehicle_type_ids

        where_sql = ' AND '.join(where_clauses)

        # Final SQL with dynamic WHERE
        query = f"""
        SELECT
            c."AgencyId" AS agency_id,
            EXTRACT(YEAR FROM COALESCE(c."PickupDate", '1970-01-01'::date)) AS year,
            EXTRACT(MONTH FROM COALESCE(c."PickupDate", '1970-01-01'::date)) AS month,
            SUM(COALESCE(c."TotalInclTax", 0)) AS revenue
        FROM "Contract" c
            LEFT JOIN "VehicleContractMap" vcm ON c."ContractId" = vcm."ContractId"
            LEFT JOIN "DefVehicle" dv ON vcm."VehicleId" = dv."VehicleId"
            LEFT JOIN "DefModel" dm ON dv."ModelId" = dm."ModelId"
            LEFT JOIN "DefVehicleCategory" dvc ON dm."VehicleCategoryId" = dvc."VehicleCategoryId"
        WHERE {where_sql}
        GROUP BY
            c."AgencyId",
            EXTRACT(YEAR FROM COALESCE(c."PickupDate", '1970-01-01'::date)),
            EXTRACT(MONTH FROM COALESCE(c."PickupDate", '1970-01-01'::date))
        ORDER BY year, month;
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params).fetchall()
                return [[row[0], int(row[1]), int(row[2]), float(row[3])] for row in result]
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}", exc_info=True)
            raise


# Initialize
contract_repository = ContractRepository(DB_URL)
forecaster = RevenueForecaster(contract_repository, look_back=6, future_months=12)


def convert_keys_to_int(obj):
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj


@app.route('/predict', methods=['POST'])
def predict_revenue():
    try:
        data = request.get_json()
        logger.info("Received payload: %s", data)

        agency_ids = data.get('agencyIds', [])
        vehicle_type_ids = data.get('vehicleTypeIds', [])
        start_date_str = data.get('startDate')
        end_date_str = data.get('endDate')

        # Validate required parameters
        if not agency_ids or not start_date_str or not end_date_str:
            logger.error("Missing required parameters: agencyIds=%s, startDate=%s, endDate=%s",
                         agency_ids, start_date_str, end_date_str)
            return jsonify({'error': 'Missing required parameters'}), 400

        # Parse dates
        try:
            start_date = parse(start_date_str)
            end_date = parse(end_date_str)
            logger.info("Parsed dates: startDate=%s, endDate=%s", start_date, end_date)
        except ValueError as e:
            logger.error("Invalid date format: %s", str(e))
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400

        # Validate IDs
        if not all(isinstance(id, int) and id > 0 for id in agency_ids):
            logger.error("Invalid agencyIds: %s", agency_ids)
            return jsonify({'error': 'All agencyIds must be positive integers'}), 400

        if vehicle_type_ids and not all(isinstance(id, int) and id > 0 for id in vehicle_type_ids):
            logger.error("Invalid vehicleTypeIds: %s", vehicle_type_ids)
            return jsonify({'error': 'All vehicleTypeIds must be positive integers'}), 400

        logger.info("Starting revenue forecasting")
        result = forecaster.forecast_revenue(agency_ids, vehicle_type_ids, start_date, end_date)

        if "error" in result:
            logger.error("Forecasting error: %s", result["error"])
            return jsonify({'error': result["error"]}), 400

        result = convert_keys_to_int(result)
        logger.info("Forecasting successful")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in predict_revenue: {str(e)}", exc_info=True)
        logger.error("Stack trace: %s", traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
