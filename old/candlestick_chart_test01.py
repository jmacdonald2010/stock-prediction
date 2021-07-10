# for learning how to make these, I'm just going to use yf data for now
# after learning how to make the charts, I will use my own remote DB data
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates
import datetime

# plot style
plt.style.use('ggplot')

symbol = "MSFT"

# gonna fetch my data from yf instead
current_date = datetime.datetime.now()
current_date = current_date.strftime("%Y-%m-%d")
sixty_days_ago = datetime.datetime.now() - datetime.timedelta(days = 59)
sixty_days_ago = sixty_days_ago.strftime("%Y-%m-%d")
data = yf.download(
    tickers = symbol,
    start = sixty_days_ago,
    end = current_date,
    interval = "15m"
)

# not sure if i have to do the convert csv to pandas df type, since I'm starting w/ a pandas df
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha='0.8')

# labels and titles
ax.set_xlabel("Date")
ax.set_ylabel("Price")
fig.suptitle("Intraday Candlestick Chart of MSFT, 15m")

# format dates
date_format = mpl_dates.DateFormatter("%d-%m-%Y %H:%M")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()