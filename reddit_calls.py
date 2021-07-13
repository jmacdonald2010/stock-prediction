import praw
import reddit_auth
import re

def get_new_symbols():
    
    # initialize the API things
    # the reddit_auth file is not on git, is used to contain API keys, etc.
    reddit = praw.Reddit(
        client_id=reddit_auth.client_id,
        client_secret=reddit_auth.client_secret,
        user_agent = reddit_auth.user_agent
    )

    popular_stocks = []
    # add to the list below as needed
    not_stocks = ['NSFW', 'OP', 'LOL', 'OMG']
    for post in reddit.subreddit("RobinhoodPennyStocks").hot(limit=20):
        comments = post.comments
        for top_level_comment in post.comments:
            if isinstance(top_level_comment, praw.models.MoreComments):
                continue
            # print(top_level_comment.body)
            words = top_level_comment.body.split()
            for word in words:
                if len(word) == 1:
                    continue
                if word.isupper():
                    word = re.sub('[^a-zA-Z]+', '', word)
                    if word in not_stocks:
                        continue
                    popular_stocks.append(word)
                elif "$" in word:
                    # word = word.replace("$", "")    # This is going to need a regex to parse these out effectively
                    word = re.sub('[^a-zA-Z]+', '', word)
                    if word in not_stocks:
                        continue
                    if word.isupper():
                        popular_stocks.append(word)

    # make a dict of the number of times certain securities are mentioned
    stock_mention_count = {}
    for stock in popular_stocks:
        if stock not in stock_mention_count:
            stock_mention_count[stock] = 1
        elif stock_mention_count[stock] >= 1:
            stock_mention_count[stock] += 1
    return stock_mention_count