# reddit_bot_2

Organization:
bot.py: praw based bot that scrapes reddit and will have the ability to post

manager.py: runs models and directs bot to do actions.

comment_success_classifier: dnn to classify comments by success. It uses ngrams, timestamps, subreddit and sentiment of the post title, the parent comment and the child/considered comment.
DNNS are not ideal for generating text but with a large db of comments most comments that would be ideal to say are already somewhere in the db.

sentiment_classifier.py: basic sentiment classifier

bot_detection_classifier.py: Adverserial network trained to detect my bot. Manager will consider this input when picking comment.