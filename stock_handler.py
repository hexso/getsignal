import sqlite3
import pandas as pd
import yfinance as yf
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "stocks_list.csv"   # CSV 파일: 종목코드, 주식명, market
DB_FILE = "stocks.db"  # SQLite DB 파일 경로

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

# --- RSI 계산 및 Telegram 메시지 전송 함수 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short=12, long=26, signal=9):
    """MACD 지표를 계산합니다.
    - macd_line: 단기 EMA와 장기 EMA의 차이
    - signal_line: macd_line의 EMA (보통 9일)
    - histogram: macd_line과 signal_line의 차이
    """
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """볼린저밴드를 계산합니다.
    - middle_band: 단순 이동평균 (window 기간)
    - upper_band: middle_band + (num_std * 표준편차)
    - lower_band: middle_band - (num_std * 표준편차)
    """
    middle_band = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = middle_band + num_std * std
    lower_band = middle_band - num_std * std
    return middle_band, upper_band, lower_band


def calculate_mfi(high, low, close, volume, window=14):
    """
    MFI (Money Flow Index)를 계산합니다.

    - typical_price: (high + low + close) / 3
    - raw_money_flow: typical_price * volume
    - 양의 머니 플로우: typical_price가 전일보다 높으면 raw_money_flow, 아니면 0
    - 음의 머니 플로우: typical_price가 전일보다 낮으면 raw_money_flow, 아니면 0
    - MFI: 100 - (100 / (1 + (rolling sum of 양의 머니 플로우 / rolling sum of 음의 머니 플로우)))

    인자:
      high: Pandas Series, 최고가
      low: Pandas Series, 최저가
      close: Pandas Series, 종가
      volume: Pandas Series, 거래량
      window: int, 기간 (기본값 14)

    반환:
      Pandas Series: 계산된 MFI 값
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    prev_typical = typical_price.shift(1)

    pos_flow = raw_money_flow.where(typical_price > prev_typical, 0)
    neg_flow = raw_money_flow.where(typical_price < prev_typical, 0)

    pos_mf = pos_flow.rolling(window=window).sum()
    neg_mf = neg_flow.rolling(window=window).sum()

    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    return mfi


def calculate_directional_indicators(high, low, close, window=14):
    """
    방향성 지표(+DI, -DI)와 ADX를 계산합니다.

    - True Range: high - low, abs(high - 이전 close), abs(low - 이전 close) 중 최댓값
    - ATR: window 기간 동안의 True Range의 단순평균
    - Up Move: 현재 high - 이전 high
    - Down Move: 이전 low - 현재 low
    - +DM: Up Move가 Down Move보다 크고 양수이면 Up Move, 그렇지 않으면 0
    - -DM: Down Move가 Up Move보다 크고 양수이면 Down Move, 그렇지 않으면 0
    - +DI: 100 * (rolling sum of +DM / ATR)
    - -DI: 100 * (rolling sum of -DM / ATR)
    - DX: 100 * |+DI - -DI| / (+DI + -DI)
    - ADX: window 기간 동안의 DX의 단순평균

    인자:
      high: Pandas Series, 최고가
      low: Pandas Series, 최저가
      close: Pandas Series, 종가
      window: int, 기간 (기본값 14)

    반환:
      tuple: (plus_di, minus_di, adx) 각 값은 Pandas Series
    """
    # True Range 계산
    high_low = high - low
    high_prev_close = (high - close.shift(1)).abs()
    low_prev_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()

    # +DM, -DM 계산
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_dm_sum = plus_dm.rolling(window=window).sum()
    minus_dm_sum = minus_dm.rolling(window=window).sum()

    plus_di = 100 * (plus_dm_sum / atr)
    minus_di = 100 * (minus_dm_sum / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()

    return plus_di, minus_di, adx

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
            columns = pd.MultiIndex.from_tuples(data.columns, names=['Price', 'Ticker'])
            data1 = pd.DataFrame(columns=columns)

            # 첫 번째 레벨의 열 이름을 소문자로 변환
            data.columns = data.columns.set_levels([data1.columns.levels[0].str.lower(), data1.columns.levels[1]])

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

def get_volume_profile(stock_code='AAPL', start_date=None, end_date=None, bins=30, draw=False):
    # 주식 데이터 다운로드
    if start_date is None:
        start_date = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    # Yahoo Finance에서 데이터 가져오기
    stock_data = yf.download(stock_code, start=start_date)

    if stock_data.empty:
        print("데이터를 가져오지 못했습니다. 날짜 범위와 종목 코드를 확인하세요.")
        return None

    # NaN 값 제거
    stock_data.dropna(inplace=True)

    # 'Price' 컬럼 생성 (고가와 저가의 평균)
    stock_data['Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3

    # 가격 구간(bin) 설정
    min_price = stock_data['Price'].min()
    max_price = stock_data['Price'].max()
    price_bins = np.linspace(min_price, max_price, bins + 1)  # bins 개수보다 1개 많아야 함

    # 매물대(Volume Profile) 계산
    volume_profile, bin_edges = np.histogram(stock_data['Price'], bins=price_bins, weights=stock_data['Volume'][stock_code])

    if draw:
        # 매물대 그래프 그리기
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(bin_edges[:-1], volume_profile, height=(bin_edges[1] - bin_edges[0]), color='blue', alpha=0.6)

        ax.set_xlabel("Volume")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"{stock_code} Volume Profile [{start_date} ~ {end_date}]")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.show()


# 예시 실행 코드
if __name__ == "__main__":
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

    get_volume_profile('AAPL')

