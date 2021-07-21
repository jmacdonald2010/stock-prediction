# based on the tutorial from realpython.com https://realpython.com/python-send-email/
import smtplib, ssl
import config
import datetime

port = 465
current_date = datetime.datetime.now().strftime('%Y-%m-%d')

# create secure SSL context
context = ssl.create_default_context()

message = f"""\
Subject: Stock Predictions, {current_date}

Good Evening,

Here is some information for the symbols on your watchlist:

(watchlist symbol stats go here)

People on Reddit are talking about these symbols:

(reddit symbols go here)

The following symbols have large projected growth, and may be worth looking into:

(other symbols go here)

And of course, this does not constitute financial/investment advice."""

with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login(config.app_email, config.app_email_pw)
    server.sendmail(config.app_email, config.email_recipient, message)