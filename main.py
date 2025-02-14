from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import os
from telegram import Bot
import pandas as pd
from stock_handler import CSV_FILE, DB_FILE, get_stock_data, calculate_rsi
import asyncio

# --- 환경 설정 ---
bot = None
user_id = None
bot_id = None

async def send_telegram_message(message: str):
    """텔레그램 메시지를 전송합니다."""
    def split_message(message, max_length=4096):
        return [message[i:i + max_length] for i in range(0, len(message), max_length)]

    messages = split_message(message)  # 긴 메시지 분할
    for msg in messages:
        await bot.send_message(chat_id=user_id, text=msg)

def start_rsi_check():
    """
    CSV 파일의 모든 종목에 대해 get_stock_data()를 호출하여 데이터를 가져온 후,
    calculate_rsi() 함수를 사용해 RSI(14일)를 계산합니다.
    RSI가 30 미만인 종목들을 '종목코드, 주식명, RSI, 최신가' 형식으로 모아
    텔레그램 메시지로 전송합니다.
    """
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].str.zfill(6)
    results = []
    for idx, row in df.iterrows():
        stock_code = str(row['종목코드']).strip()
        stock_name = str(row['주식명']).strip()
        market = str(row['market']).strip()
        data = get_stock_data(stock_code, market, DB_FILE)
        if data is None or data.empty:
            continue
        # calculate_rsi 함수 활용
        rsi_series = calculate_rsi(data['close'])
        latest_rsi = rsi_series.iloc[-1]
        latest_price = data['close'].iloc[-1]
        if latest_rsi < 30:
            results.append(f"{stock_code} {stock_name}: RSI {latest_rsi:.2f}, Price {latest_price:.2f}")
    if results:
        message_text = "매수 신호 종목 (RSI < 30):\n" + "\n".join(results)
    else:
        message_text = "현재 RSI가 30 미만인 종목이 없습니다."
    asyncio.run(send_telegram_message(message_text))
    return {"message": "RSI 체크 완료", "result": message_text}


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    # 파일을 읽고 내용을 변수에 저장하는 코드
    with open('private.txt', 'r') as file:
        # 파일 내용을 줄 단위로 읽어옴
        lines = file.readlines()

    for line in lines:
        if line.startswith('token:'):
            bot_token = line.split(': ')[1].strip()  # ':' 이후 값에서 공백을 제거하여 저장
            bot_id = line.split(':')[1].strip()
        elif line.startswith('id:'):
            user_id = line.split(':')[1].strip()  # ':' 이후 값에서 공백을 제거하여 저장

    bot = Bot(bot_token)

    result = start_rsi_check()
    print(result)