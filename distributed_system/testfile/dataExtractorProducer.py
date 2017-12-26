from kombu.connection import Connection
from kombu import Producer
from time import sleep
from configurations.config import settings
from dataExtraction import tweetDataExtraction
# get the consumer setting
data = settings()
connection,exchange,task_queue1,task_queue2,task_queue3 = data.get_producer_settings()
publisher = Producer(connection,exchange=exchange)
# get all the tweets
data_extraction = tweetDataExtraction()
tweets = data_extraction.getData()

for tweet in tweets:
    print("Data")
    # Mongo Queue
    #publisher.publish(tweet,routing_key=task_queue1.routing_key)
    
    # classifier queue
    publisher.publish(tweet,routing_key=task_queue2.routing_key)
    
    # Sql queue
    # Specify that the data is the tweet_info
    #tweet['data_type'] = "tweet_info"
    #publisher.publish(tweet,routing_key=task_queue3.routing_key)
    
    sleep(.05)
publisher.close()
connection.close()
