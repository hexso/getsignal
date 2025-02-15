import sqlite3
import pandas as pd
import datetime
from stock_handler import CSV_FILE, DB_FILE, get_stock_data, calculate_rsi, update_stock_data_from_csv, calculate_macd, calculate_bollinger_bands


# 백테스트용 함수: 하나의 종목에 대해 전략을 시뮬레이션
def backtest_stock(ticker, db_path):
    # DB에서 해당 티커의 데이터를 날짜 순으로 불러옴
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM stock_data WHERE ticker=? ORDER BY date", conn, params=(ticker,))
    conn.close()
    if df.empty:
        return []

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # 'close' 컬럼을 사용하여 지표 계산 (컬럼명이 DB에 저장된 그대로여야 합니다)
    close_series = df['close']
    open_series = df['open']

    # RSI(14일) 계산
    rsi_series = calculate_rsi(close_series)
    # 볼린저밴드 계산 (여기서는 middle, upper, lower 밴드 반환)
    _, _, lower_band = calculate_bollinger_bands(close_series)

    trades = []
    # i번째 날에 조건이 만족되면, i번째 날의 종가에 매수하고, i+1번째 날의 종가에 매도하는 거래를 시뮬레이션
    for i in range(len(df) - 1):
        if pd.isna(rsi_series.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        if rsi_series.iloc[i] < 30 and close_series.iloc[i] <= lower_band.iloc[i]:
            buy_price = close_series.iloc[i]
            sell_price = open_series.iloc[i + 1]
            ret = (sell_price - buy_price) / buy_price  # 단순 수익률 계산
            trade_date = df.index[i].strftime("%Y-%m-%d")
            trades.append({
                "date": trade_date,
                "ticker": ticker,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "return": ret
            })
    return trades


# 여러 종목에 대해 백테스트를 수행하고 결과를 집계하는 함수
def backtest_strategy(csv_filename, db_path):
    # CSV 파일에는 종목코드, 주식명, market 컬럼이 있어야 합니다.
    df = pd.read_csv(csv_filename, encoding="utf-8-sig", dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].str.zfill(6)

    all_trades = []
    for idx, row in df.iterrows():
        # 티커 구성: 예) "005930.KS"
        ticker = f"{row['종목코드']}.{row['market'].strip()}"
        trades = backtest_stock(ticker, db_path)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        print("전략 조건에 맞는 거래가 실행되지 않았습니다.")
        return trades_df

    # 전체 거래 건수와 평균 수익률 계산
    total_trades = len(trades_df)
    avg_return = trades_df['return'].mean()
    print(f"총 거래 건수: {total_trades}, 평균 수익률: {avg_return:.2%}")
    print(trades_df)
    return trades_df


# 예시 실행 코드
if __name__ == "__main__":
    # CSV 파일과 DB 파일 경로를 지정합니다.
    trades_df = backtest_strategy(CSV_FILE, DB_FILE)
