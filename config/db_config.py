# config/db_config.py
# 이 파일은 데이터베이스에 연결하기 위한 설정 정보를 저장합니다.

DATABASE = {
    # 데이터베이스 사용자 계정 정보
    'user': 'postgres',         # 데이터베이스 사용자 이름
    'password': '1234',         # 데이터베이스 사용자 비밀번호

    # 데이터베이스 서버 정보
    'host': 'localhost',        # 서버 주소 (localhost는 현재 컴퓨터를 의미)
    'port': 5432,               # 서버 포트 번호 (PostgreSQL의 기본 포트)

    # 데이터베이스 이름
    'dbname': 'my_trading_db'   # 연결할 데이터베이스의 이름
}
