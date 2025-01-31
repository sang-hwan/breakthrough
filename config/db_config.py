# config/db_config.py
# 이 파일은 데이터베이스에 연결하기 위한 설정 정보를 간단히 정리해둔 곳입니다.
# 아래의 'DATABASE' 딕셔너리는 PostgreSQL 접속에 필요한 정보를 담고 있습니다.

DATABASE = {
    'user': 'postgres',        # DB에 접속할 계정 이름
    'password': '1234',        # 해당 계정의 비밀번호
    'host': 'localhost',       # DB 서버 주소(현재 PC라면 localhost)
    'port': 5432,              # PostgreSQL 기본 포트번호
    'dbname': 'my_trading_db'  # 실제 사용할 데이터베이스 이름
}
