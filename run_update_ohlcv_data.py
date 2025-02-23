# run_update_ohlcv_data.py
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

from data.db.db_config import DATABASE
from data.db.db_manager import fetch_ohlcv_records, insert_ohlcv_records
from data.ohlcv.ohlcv_fetcher import fetch_historical_ohlcv_data, get_top_volume_symbols, get_latest_onboard_date
from logs.logger_config import initialize_root_logger, setup_logger

load_dotenv()
initialize_root_logger()
logger = setup_logger(__name__)

def create_database_if_not_exists(db_config):
    """
    .env에 명시된 DB가 존재하지 않으면, postgres 기본 DB에 접속하여 해당 DB를 생성합니다.
    """
    dbname = db_config.get('dbname')
    user = db_config.get('user')
    password = db_config.get('password')
    host = db_config.get('host')
    port = db_config.get('port')
    try:
        conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        exists = cur.fetchone()
        if not exists:
            logger.debug(f"Database '{dbname}' does not exist. Creating database.")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        else:
            logger.debug(f"Database '{dbname}' already exists.")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error checking/creating database: {e}", exc_info=True)
        sys.exit(1)

def run_update_ohlcv_data():
    create_database_if_not_exists(DATABASE)
    
    # get_top_volume_symbols now returns a list of tuples: (symbol, onboard_date)
    symbols_with_onboard = get_top_volume_symbols(exchange_id='binance', quote_currency='USDT', count=5)
    if not symbols_with_onboard:
        logger.error("No valid symbols found from Binance.")
        sys.exit(1)
    logger.debug(f"Top symbols (with onboardDate): {symbols_with_onboard}")
    
    # For further processing, we extract just the symbol names for table naming
    symbols = [item[0] if isinstance(item, tuple) else item for item in symbols_with_onboard]
    
    timeframes = ["1d", "4h", "1h", "15m"]
    
    # get_latest_onboard_date now accepts symbols in either form.
    global_start_date = get_latest_onboard_date(symbols, exchange_id='binance')
    logger.debug(f"Unified start date for all symbols: {global_start_date}")
    
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    for symbol in symbols:
        symbol_key = symbol.replace("/", "").lower()
        logger.debug(f"Processing {symbol} (table prefix: ohlcv_{symbol_key}_)")
        for tf in timeframes:
            table_name = f"ohlcv_{symbol_key}_{tf}"
            logger.debug(f"Processing {symbol} - {tf} (table: {table_name})")
            
            try:
                df_existing = fetch_ohlcv_records(table_name=table_name)
            except Exception as e:
                logger.error(f"Error fetching existing data for table {table_name}: {e}", exc_info=True)
                df_existing = pd.DataFrame()
            
            if not df_existing.empty:
                last_timestamp = df_existing.index.max()
                new_start_dt = last_timestamp + timedelta(seconds=1)
                new_start_date = new_start_dt.strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"Existing data found in {table_name}. Fetching new data from {new_start_date} to {end_date}.")
            else:
                new_start_date = global_start_date
                logger.debug(f"No existing data in {table_name}. Fetching data from {new_start_date} to {end_date}.")
            
            try:
                df_new = fetch_historical_ohlcv_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=new_start_date,
                    exchange_id='binance'
                )
                if df_new.empty:
                    logger.debug(f"No new data fetched for {symbol} - {tf}.")
                    continue
                else:
                    logger.debug(f"Fetched {len(df_new)} new rows for {symbol} - {tf}.")
            except Exception as e:
                logger.error(f"Error fetching OHLCV data for {symbol} - {tf}: {e}", exc_info=True)
                continue
            
            try:
                insert_ohlcv_records(df_new, table_name=table_name)
                logger.debug(f"Inserted new data into table {table_name}.")
            except Exception as e:
                logger.error(f"Error inserting data into table {table_name}: {e}", exc_info=True)

if __name__ == "__main__":
    run_update_ohlcv_data()
