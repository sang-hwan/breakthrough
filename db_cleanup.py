# db_cleanup.py

from sqlalchemy import create_engine, MetaData
from config.db_config import DATABASE

def drop_all_tables():
    db_config = DATABASE
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    meta = MetaData()
    meta.reflect(bind=engine)
    
    if meta.tables:
        meta.drop_all(engine)
        print("모든 테이블이 삭제되었습니다.")
    else:
        print("데이터베이스에 삭제할 테이블이 없습니다.")

if __name__ == "__main__":
    drop_all_tables()
