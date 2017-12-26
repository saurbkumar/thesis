from kombu.mixins import ConsumerMixin
from configurations.config import settings
from dataInsert import mongoDataInsert
from time import gmtime, strftime


# MongoDb consumer - store data in mongodb

class mongoConsumer(ConsumerMixin):

    def __init__(self, connection,queue,obj):
        self.queue = queue
        self.connection = connection
        self.tweet_insert = obj
    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=[self.queue],callbacks=[self.on_task])]
    
    def on_consume_ready(connection, channel, consumers,d):
        print("Mongo Consumer Ready")
    def on_task(self, body, message):
        print("Data Status for Mongo Queue at {} is ".format(strftime("%Y-%m-%d %H:%M:%S",gmtime()))),
        try:
            self.tweet_insert.insert(body)
            print("Record Inserted")
        except:
            print("Data is not inserted and current packet is ignored")
        finally:
            # acknowledge the message to remove it form the queue
            message.ack()
def runConsumer():
    obj = mongoDataInsert()
    data = settings()
    connection,queue, = data.get_mongo_queue_setting()
    mongoConsumer(connection,queue,obj).run()
    consumer.close()
    connection.close()