import sqlite3
from os import close

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
    high_series = df['high']

    # RSI(14일) 계산
    rsi_series = calculate_rsi(close_series)
    rsi_series_5 = calculate_rsi(close_series, 5)
    # 볼린저밴드 계산 (여기서는 middle, upper, lower 밴드 반환)
    _, _, lower_band = calculate_bollinger_bands(close_series)

    #MACD 계산
    macd_series, signal_series, _ = calculate_macd(close_series)


    trades = []
    for i in range(2, len(df) - 1):
        if pd.isna(rsi_series.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        # 매수 조건: RSI와 5일 RSI 모두 30 미만이며, 전환점을 형성하는지 확인
        if (rsi_series.iloc[i] < 30 and rsi_series_5.iloc[i] < 30 and
                (rsi_series.iloc[i - 2] > rsi_series.iloc[i - 1] < rsi_series.iloc[i]) and
                (rsi_series_5.iloc[i - 2] > rsi_series_5.iloc[i - 1] < rsi_series_5.iloc[i])):

            buy_price = close_series.iloc[i]
            sell_price = None
            sell_day = None

            # 매수 후 다음 10일 동안 조건 확인
            for j in range(i + 1, min(i + 11, len(df))):
                current_rsi = rsi_series.iloc[j]
                current_return = (high_series.iloc[j] - buy_price) / buy_price  # 당일 고가 기준 수익률
                # 조건: RSI가 50 초과 또는 수익률이 10% 이상이면 매도
                if current_rsi > 50 :#or current_return >= 0.10:
                    # 목표 수익률 조건 충족 시에는 10% 목표 가격로 매도, 그렇지 않으면 당일 종가로 매도
                    # if current_return >= 0.10:
                    #     sell_price = buy_price * 1.10
                    # else:
                    sell_price = close_series.iloc[j]
                    sell_day = j
                    break

            # 10일 내 조건 미충족 시, 10일째 날 매도
            if sell_price is None:
                sell_day = min(i + 10, len(df) - 1)
                sell_price = close_series.iloc[sell_day]

            ret = (sell_price - buy_price) / buy_price  # 단순 수익률 계산
            trades.append({
                "buy_date": df.index[i].strftime("%Y-%m-%d"),
                "sell_date": df.index[sell_day].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "return": ret
            })

    return trades


# 백테스트용 함수: 급락한것을 매수해보자.
def backtest_stock2(ticker, db_path):
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
    high_series = df['high']

    # RSI(14일) 계산
    rsi_series = calculate_rsi(close_series)
    rsi_series_5 = calculate_rsi(close_series, 5)
    # 볼린저밴드 계산 (여기서는 middle, upper, lower 밴드 반환)
    _, _, lower_band = calculate_bollinger_bands(close_series)

    #MACD 계산
    macd_series, signal_series, _ = calculate_macd(close_series)


    trades = []
    for i in range(2, len(df) - 1):
        if pd.isna(rsi_series.iloc[i]) or pd.isna(lower_band.iloc[i]):
            continue
        # 매수 조건: RSI와 5일 RSI 모두 30 미만이며, 전환점을 형성하는지 확인
        if (rsi_series.iloc[i] < 30 and rsi_series_5.iloc[i] < 30 and
                (rsi_series.iloc[i - 2] > rsi_series.iloc[i - 1] < rsi_series.iloc[i]) and
                (rsi_series_5.iloc[i - 2] > rsi_series_5.iloc[i - 1] < rsi_series_5.iloc[i])):

            buy_price = close_series.iloc[i]
            sell_price = None
            sell_day = None

            # 매수 후 다음 10일 동안 조건 확인
            for j in range(i + 1, min(i + 11, len(df))):
                current_rsi = rsi_series.iloc[j]
                current_return = (high_series.iloc[j] - buy_price) / buy_price  # 당일 고가 기준 수익률
                # 조건: RSI가 50 초과 또는 수익률이 10% 이상이면 매도
                if current_rsi > 50 :#or current_return >= 0.10:
                    # 목표 수익률 조건 충족 시에는 10% 목표 가격로 매도, 그렇지 않으면 당일 종가로 매도
                    # if current_return >= 0.10:
                    #     sell_price = buy_price * 1.10
                    # else:
                    sell_price = close_series.iloc[j]
                    sell_day = j
                    break

            # 10일 내 조건 미충족 시, 10일째 날 매도
            if sell_price is None:
                sell_day = min(i + 10, len(df) - 1)
                sell_price = close_series.iloc[sell_day]

            ret = (sell_price - buy_price) / buy_price  # 단순 수익률 계산
            trades.append({
                "buy_date": df.index[i].strftime("%Y-%m-%d"),
                "sell_date": df.index[sell_day].strftime("%Y-%m-%d"),
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
