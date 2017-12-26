from kombu.mixins import ConsumerMixin
from configurations.config import settings
from SQLinsert import mysqlInsert
from time import gmtime, strftime


# SQL consumer - store data in mongodb

class sqlConsumer(ConsumerMixin):

    def __init__(self, connection,queue,obj):
        self.queue = queue
        self.connection = connection
        self.tweet_insert = obj
    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=[self.queue],callbacks=[self.on_task])]
    
    def on_consume_ready(connection, channel, consumers,d):
        print("SQL Consumer Ready")
    def on_task(self, body, message):
        print("Data Status for SQL Queue at {} is ".format(strftime("%Y-%m-%d %H:%M:%S",gmtime()))),
        try:
            # inser function is of three type
            '''
            insertData - used to insert tweet info
            insertclassifrInfo - used to update the classifier  info
            inserLocationInfo - used to update the location  info 
            '''
            self.tweet_insert.insert(body)
        except:
            print("Data is not inserted due to connectino error or something else, check error traceback in SQL consumer")
        finally:
            # acknowledge the message to remove it form the queue
            message.ack()
def runConsumer():
    obj = mysqlInsert()
    data = settings()
    connection,queue, = data.get_sql_queue_setting()
    sqlConsumer(connection,queue,obj).run()
    consumer.close()
    connection.close()