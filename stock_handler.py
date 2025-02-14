import sqlite3
import pandas as pd
import yfinance as yf
import datetime
import os


def create_db_table(conn):
    """
    주식 데이터를 저장할 테이블을 생성합니다.
    stock_name 컬럼을 추가하여 주식명도 저장합니다.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        stock_name TEXT,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        UNIQUE(ticker, date)
    );
    """
    conn.execute(create_table_query)
    conn.commit()


# 1. CSV 파일에 있는 종목목록을 읽어 지난 1년치 데이터를 DB에 저장하는 함수
def store_stock_data_from_csv(csv_filename, db_path):
    # CSV 파일은 종목코드, 주식명, market 순으로 저장되어 있음
    # 종목코드를 문자열로 읽고, 6자리(앞의 0 포함)로 변환
    df = pd.read_csv(csv_filename, encoding="utf-8-sig", dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].str.zfill(6)

    conn = sqlite3.connect(db_path)
    create_db_table(conn)

    for idx, row in df.iterrows():
        ticker = str(row['종목코드']).strip() + "." + str(row['market']).strip()
        stock_name = str(row['주식명']).strip()
        print(f"{ticker} ({stock_name}) 데이터 저장 시작...")
        try:
            # 지난 1년치 일별 데이터 다운로드
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
            if data.empty:
                print(f"{ticker}: 데이터가 없습니다.")
                continue

            # 인덱스를 컬럼으로 변환 후 "Date" 컬럼을 datetime으로 강제 변환
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])

            for _, d in data.iterrows():
                # pd.to_datetime()로 변환한 후 strftime 호출
                date_str = str(pd.to_datetime(d['Date'].iloc[0]).strftime("%Y-%m-%d"))
                insert_query = """
                INSERT OR IGNORE INTO stock_data (ticker, stock_name, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(insert_query, (
                    ticker,
                    stock_name,
                    date_str,
                    float(d['Open'].iloc[0]),
                    float(d['High'].iloc[0]),
                    float(d['Low'].iloc[0]),
                    float(d['Close'].iloc[0]),
                    int(d['Volume'].iloc[0])  # 이미 올바르게 처리됨
                ))

            conn.commit()
            print(f"{ticker} ({stock_name}) 데이터 저장 완료.")
        except Exception as e:
            print(f"{ticker} 데이터 저장 중 에러 발생: {e}")

    conn.close()


def update_stock_data_from_csv(csv_filename, db_path):
    # CSV 파일은 종목코드, 주식명, market 순으로 저장되어 있음
    df = pd.read_csv(csv_filename, encoding="utf-8-sig", dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].str.zfill(6)

    conn = sqlite3.connect(db_path)
    create_db_table(conn)
    cur = conn.cursor()

    for idx, row in df.iterrows():
        ticker = str(row['종목코드']).strip() + "." + str(row['market']).strip()
        stock_name = str(row['주식명']).strip()
        # DB에서 해당 티커의 마지막 저장된 날짜 조회
        cur.execute("SELECT MAX(date) FROM stock_data WHERE ticker = ?", (ticker,))
        result = cur.fetchone()
        last_date_str = result[0]
        if last_date_str:
            last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d")
            start_date = last_date + datetime.timedelta(days=1)
        else:
            start_date = datetime.datetime.today() - datetime.timedelta(days=365)

        end_date = datetime.datetime.today()

        # 업데이트할 데이터가 없으면 건너뜁니다.
        if start_date.date() > end_date.date():
            print(f"{ticker}: 이미 최신 데이터가 저장되어 있습니다.")
            continue

        print(f"{ticker} ({stock_name}) 데이터 최신화 시작: {start_date.date()} ~ {end_date.date()}")
        try:
            data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                               end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                               interval="1d", progress=False)
            if data.empty:
                print(f"{ticker}: 업데이트할 데이터가 없습니다.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])

            for _, d in data.iterrows():
                date_str = str(pd.to_datetime(d['Date']).strftime("%Y-%m-%d"))
                insert_query = """
                INSERT OR IGNORE INTO stock_data (ticker, stock_name, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                conn.execute(insert_query, (
                    ticker,
                    stock_name,
                    date_str,
                    float(d['Open']),
                    float(d['High']),
                    float(d['Low']),
                    float(d['Close']),
                    int(d['Volume'])
                ))
            conn.commit()
            print(f"{ticker} ({stock_name}) 데이터 최신화 완료.")
        except Exception as e:
            print(f"{ticker} 최신화 중 에러: {e}")

    conn.close()

def get_stock_data(stock_code, market, db_path):
    """
    주어진 종목코드와 market을 이용해 ticker를 구성합니다.
    현재 시간이 주식시장 마감(오후 4시) 전이면 yfinance API를 통해 최신 데이터를,
    마감 후이면 DB에 저장된 데이터를 불러옵니다.
    """
    # 숫자로 처리되어 앞의 0이 사라지는 문제를 방지하기 위해 6자리 문자열로 변환
    stock_code = str(stock_code).strip().zfill(6)
    ticker = stock_code + "." + str(market).strip()
    now = datetime.datetime.now()
    market_close = now.replace(hour=16, minute=30, second=0, microsecond=0)

    if now < market_close:
        print(f"{ticker}: 시장 마감 전. API를 통해 최신 데이터를 가져옵니다.")
        try:
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
            return data
        except Exception as e:
            print(f"{ticker} API 데이터 호출 중 에러: {e}")
            return None
    else:
        print(f"{ticker}: 시장 마감 후. DB에 저장된 데이터를 불러옵니다.")
        try:
            conn = sqlite3.connect(db_path)
            query = "SELECT * FROM stock_data WHERE ticker = ? ORDER BY date"
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            if df.empty:
                print(f"{ticker}: DB에 저장된 데이터가 없습니다.")
                return None
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"{ticker} DB 데이터 호출 중 에러: {e}")
            return None

# 예시 실행 코드
if __name__ == "__main__":
    CSV_FILE = "stocks_list.csv"  # CSV 파일 경로 (종목코드, 주식명, market)
    DB_FILE = "stocks.db"  # SQLite DB 파일 경로

    # 1. 초기 저장: DB가 없으면 CSV에 있는 종목들의 지난 1년치 데이터를 DB에 저장
    if not os.path.exists(DB_FILE):
        print("DB가 존재하지 않으므로 초기 저장을 진행합니다.")
        store_stock_data_from_csv(CSV_FILE, DB_FILE)
    else:
        print("DB 파일이 이미 존재합니다. 초기 저장은 건너뜁니다.")

    # 2. 최신화 함수는 매일 오후 4시 20분 이후에 실행되도록 스케줄러와 연동하면 됩니다.
    #    (여기서는 수동 호출 예시)
    #update_stock_data_from_csv(CSV_FILE, DB_FILE)

    # 3. 종목정보 조회 예시:
    # 예를 들어, 종목코드 "005930"과 market "KS"를 입력받은 경우:
    stock_info = get_stock_data("005930", "KS", DB_FILE)
    if stock_info is not None:
        print(stock_info.head())
