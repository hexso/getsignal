from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import pandas as pd
from stock_handler import CSV_FILE, DB_FILE, get_stock_data, calculate_rsi, update_stock_data_from_csv, calculate_macd, \
    calculate_bollinger_bands, US_CSV_FILE
import asyncio
import logging
import nest_asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- 환경 설정 ---
bot = None
user_id = None
bot_id = None
bot_token = None
RSI_LOWER = 20
TRADE_AMOUNT = 100000000 #거래금액 1억이상

def telegram_init():
    global bot_token, bot_id, user_id, bot
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

async def send_telegram_message(message: str):
    """텔레그램 메시지를 전송합니다."""
    def split_message(message, max_length=4096):
        return [message[i:i + max_length] for i in range(0, len(message), max_length)]

    messages = split_message(message)  # 긴 메시지 분할
    for msg in messages:
        await bot.send_message(chat_id=user_id, text=msg)

# 텔레그램 메시지 핸들러: "start" 메시지를 받으면 start_rsi_check() 함수를 실행
async def handle_start_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    if text == "start":
        # 동기 함수를 blocking 없이 실행 (필요한 경우)
        result = await asyncio.to_thread(start_kr_market)
        # result는 {"message": ..., "result": ...} 형식의 dict라고 가정
        await update.message.reply_text("start에 대한 명령을 완료하였습니다.")
        await send_telegram_message(result["result"])
    elif text == "us":
        # 동기 함수를 blocking 없이 실행 (필요한 경우)
        result = await asyncio.to_thread(start_us_market)
        # result는 {"message": ..., "result": ...} 형식의 dict라고 가정
        await update.message.reply_text("us 대한 명령을 완료하였습니다.")
        await send_telegram_message(result["result"])
    else:
        await update.message.reply_text("알 수 없는 명령입니다. 'start'를 보내주세요.")


def start_kr_market():
    """
    CSV 파일의 모든 종목에 대해 get_stock_data()를 호출하여 데이터를 가져온 후,
    calculate_rsi()를 통해 RSI(14일), calculate_macd()를 통해 MACD,
    calculate_bollinger_bands()를 통해 볼린저밴드를 계산합니다.
    RSI가 30 미만인 종목에 대해 '종목코드, 주식명, RSI, 최신가, MACD(및 Signal), 볼린저밴드(Upper, Middle, Lower)'
    정보를 포맷팅하여 results에 저장한 후, 텔레그램 메시지로 전송합니다.
    """
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype={'종목코드': str})
    df['종목코드'] = df['종목코드'].str.zfill(6)
    results = []
    for idx, row in df.iterrows():
        stock_code = str(row['종목코드']).strip()
        stock_name = str(row['주식명']).strip()
        market = str(row['market']).strip()
        ticker = stock_code + "." + str(market).strip()
        data = get_stock_data(stock_code, ko_market=market,market="KR")
        if data is None or data.empty:
            continue

        # RSI 계산 (calculate_rsi 함수는 이미 구현되어 있다고 가정)
        rsi_series = calculate_rsi(data['close'])
        latest_rsi = rsi_series.iloc[-1]
        rsi_yesterday = rsi_series.iloc[-2]
        rsi_before_yesterday = rsi_series.iloc[-2]
        latest_price = data['close'].iloc[-1]
        latest_volume = data['volume'].iloc[-1]

        # MACD 계산
        macd_line, signal_line, _ = calculate_macd(data['close'])


        # RSI가 20미만, RSI가 추세전환 했는지 확인
        if latest_rsi < RSI_LOWER and rsi_yesterday > latest_rsi < rsi_before_yesterday and latest_price * latest_volume > TRADE_AMOUNT :# and latest_before_rsi < latest_rsi:
            results.append(
                f"{stock_code} {stock_name}:\n"
                f"  RSI: {latest_rsi:.2f}, Price: {latest_price:.2f}\n"
            )
            print(f'{stock_code}의 RSI가 {RSI_LOWER}이하입니다.  {latest_rsi}')

    if results:
        message_text = "매수 신호 종목:\n" + "\n\n".join(results)
    else:
        message_text = "현재 조건에 맞는 종목이 없습니다."

    return {"message": "체크 완료", "result": message_text}


def start_us_market():
    """
    CSV 파일의 모든 종목에 대해 get_stock_data()를 호출하여 데이터를 가져온 후,
    calculate_rsi()를 통해 RSI(14일), calculate_macd()를 통해 MACD,
    calculate_bollinger_bands()를 통해 볼린저밴드를 계산합니다.
    RSI가 30 미만인 종목에 대해 '종목코드, 주식명, RSI, 최신가, MACD(및 Signal), 볼린저밴드(Upper, Middle, Lower)'
    정보를 포맷팅하여 results에 저장한 후, 텔레그램 메시지로 전송합니다.
    """
    df = pd.read_csv(US_CSV_FILE, encoding="utf-8-sig", dtype={'종목코드': str})
    results = []
    for idx, row in df.iterrows():
        stock_code = str(row['종목코드']).strip()
        stock_name = str(row['주식명']).strip()
        ticker = stock_code
        data = get_stock_data(stock_code, market="US")
        if data is None or data.empty:
            continue

        # RSI 계산 (calculate_rsi 함수는 이미 구현되어 있다고 가정)
        rsi_series = calculate_rsi(data['close'])
        latest_rsi = rsi_series.iloc[-1][ticker]
        rsi_yesterday = rsi_series.iloc[-2][ticker]
        rsi_before_yesterday = rsi_series.iloc[-2][ticker]
        latest_price = data['close'].iloc[-1][ticker]
        latest_volume = data['volume'].iloc[-1][ticker]

        # MACD 계산
        macd_line, signal_line, _ = calculate_macd(data['close'])


        # RSI가 20미만, RSI가 추세전환 했는지 확인
        if latest_rsi < RSI_LOWER and rsi_yesterday > latest_rsi < rsi_before_yesterday and latest_price * latest_volume > TRADE_AMOUNT :# and latest_before_rsi < latest_rsi:
            results.append(
                f"{stock_code} {stock_name}:\n"
                f"  RSI: {latest_rsi:.2f}, Price: {latest_price:.2f}\n"
            )
            print(f'{stock_code}의 RSI가 {RSI_LOWER}이하입니다.  {latest_rsi}')

    if results:
        message_text = "매수 신호 종목:\n" + "\n\n".join(results)
    else:
        message_text = "현재 조건에 맞는 종목이 없습니다."

    return {"message": "체크 완료", "result": message_text}

# 스케줄러에서 호출할 함수: 매일 오후 4시 30분에 실행되어 DB를 최신화
def scheduled_kr_stock_update():
    print("스케줄러: 한국 주식 데이터 최신화 시작")
    update_stock_data_from_csv("KR")
    print("스케줄러: 한국 주식 데이터 최신화 완료")

def scheduled_us_stock_update():
    print("스케줄러: 미국 주식 데이터 최신화 시작")
    update_stock_data_from_csv("US")
    print("스케줄러: 미국 주식 데이터 최신화 완료")

async def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_start_message))
    # APScheduler를 이용해 매일 오후 4시 30분에 scheduled_stock_update() 실행
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_kr_stock_update, "cron", day_of_week="mon-fri", hour=16, minute=30)
    scheduler.add_job(scheduled_us_stock_update, "cron", day_of_week="tue-sat", hour=7, minute=0)
    scheduler.start()

    await app.run_polling(close_loop=False)


if __name__ == "__main__":
    scheduled_kr_stock_update()
    telegram_init()
    nest_asyncio.apply()
    asyncio.run(main())
    #scheduled_stock_update()
    #result = start_kr_market()
    #print(result)