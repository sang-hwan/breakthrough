# data/deleter/delete_data.py
"""
이 스크립트는 데이터베이스의 모든 테이블을 삭제하는 기능을 수행합니다.
주로 테스트 환경에서 초기화를 위해 사용될 수 있으며, 
PostgreSQL의 public 스키마 내 모든 테이블을 제거합니다.
"""

import sys  # 시스템 종료 및 예외 처리를 위해 사용
from dotenv import load_dotenv  # 환경변수 로드를 위한 라이브러리
from sqlalchemy import create_engine, text  # 데이터베이스 연결 및 SQL 실행을 위한 라이브러리
from data.db.db_config import DATABASE  # 데이터베이스 접속 정보를 담은 설정 객체
from logs.log_config import initialize_root_logger, setup_logger  # 로깅 초기화 및 설정 함수

# 환경변수 로드 및 로깅 초기화
load_dotenv()  # .env 파일에 정의된 환경변수를 시스템에 로드합니다.
initialize_root_logger()  # 루트 로거를 초기화합니다.
logger = setup_logger(__name__)  # 모듈 별 로거를 생성합니다.

def drop_all_tables(db_config):
    """
    데이터베이스 내의 모든 테이블을 삭제합니다.
    
    Parameters:
        db_config (dict): 데이터베이스 접속 정보를 담은 딕셔너리 
                          (예: {'user': ..., 'password': ..., 'host': ..., 'port': ..., 'dbname': ...})
    
    Returns:
        None: 모든 테이블 삭제 작업 후 성공 메시지 혹은 에러 로그를 남깁니다.
    """
    try:
        # 데이터베이스 연결 문자열을 구성하고 SQLAlchemy 엔진을 생성합니다.
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True  # 커넥션의 유효성을 주기적으로 확인합니다.
        )
        # 데이터베이스 트랜잭션을 시작합니다. (자동 커밋 모드)
        with engine.begin() as conn:
            # public 스키마 내의 모든 테이블 이름을 조회하는 SQL 문 실행
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]  # 결과에서 테이블 이름만 추출하여 리스트로 생성
            if not tables:
                logger.debug("No tables found in the database.")
                return
            # 조회된 각 테이블에 대해 DROP TABLE 명령어를 실행합니다.
            for table in tables:
                logger.debug(f"Dropping table {table}...")
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            logger.debug("All tables dropped successfully.")
    except Exception as e:
        # 예외 발생 시 에러 로그를 기록하고 프로그램을 종료합니다.
        logger.error(f"Error dropping tables: {e}", exc_info=True)
        sys.exit(1)

def run_drop_db_tables():
    """
    데이터베이스 접속 정보(DATABASE)를 사용하여 drop_all_tables 함수를 실행합니다.
    
    Parameters:
        없음
    
    Returns:
        None
    """
    drop_all_tables(DATABASE)

# 스크립트가 직접 실행될 경우 run_drop_db_tables 함수를 호출합니다.
if __name__ == "__main__":
    run_drop_db_tables()
