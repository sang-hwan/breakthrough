# data/db/db_config.py

# 전역 변수 DATABASE:
# 이 변수는 데이터베이스 접속에 필요한 정보를 담고 있습니다.
# PostgreSQL 데이터베이스와 연결하기 위해 사용자 이름, 비밀번호, 호스트, 포트, 데이터베이스 이름을 설정합니다.
DATABASE: dict = {
    'user': 'postgres',      # 데이터베이스 접속에 사용되는 사용자 이름입니다.
    'password': '1234',      # 사용자의 비밀번호입니다.
    'host': 'localhost',     # 데이터베이스가 설치된 서버 주소입니다. 여기서는 로컬 서버를 사용합니다.
    'port': 5432,            # PostgreSQL의 기본 포트 번호입니다.
    'dbname': 'my_trading_db'  # 접속할 데이터베이스의 이름입니다.
}
