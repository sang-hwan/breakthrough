# data/db/db_config.py
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE: dict = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'dbname': os.getenv('DB_NAME')
}
