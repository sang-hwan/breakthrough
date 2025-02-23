# run_drop_db_tables.py
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from data.db.db_config import DATABASE
from logs.logger_config import initialize_root_logger, setup_logger

# 환경변수 로드 및 로깅 초기화
load_dotenv()
initialize_root_logger()
logger = setup_logger(__name__)

def drop_all_tables(db_config):
    try:
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
        with engine.begin() as conn:
            # public 스키마 내 테이블 목록 조회
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]
            if not tables:
                logger.debug("No tables found in the database.")
                return
            for table in tables:
                logger.debug(f"Dropping table {table}...")
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            logger.debug("All tables dropped successfully.")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}", exc_info=True)
        sys.exit(1)

def run_drop_db_tables():
    drop_all_tables(DATABASE)

if __name__ == "__main__":
    run_drop_db_tables()
