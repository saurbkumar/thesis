from kombu.mixins import ConsumerMixin
from configurations.config import settings
from classifierModel import TweetClassifier
from kombu import Producer
from time import sleep
import sys
from time import gmtime, strftime

# Classifier consumer

class classifierConsumer(ConsumerMixin):

    def __init__(self, connection,queue,obj,publisher,sql_queue):
        # 
        self.queue = queue
        self.connection = connection
        # obj is instance of the classifier
        self.obj = obj
        # publisher to insert data in to sql queue
        self.publisher = publisher
        # SQL queue
        self.task_queue3 = sql_queue

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=[self.queue],callbacks=[self.on_task])]
    
    def on_consume_ready(connection, channel, consumers,d):
        print("Classifier Consumer Ready")
        
    def on_task(self, body, message):
        print("Data Status for Classifier Queue at {} is ".format(strftime("%Y-%m-%d %H:%M:%S",gmtime()))),
        try:
            result = self.obj.classify(body['text'])
            # insert this data in  to the mysql queue
            # augmenting current data
            body['data_type'] = 'classifier_info'
            # to list as np arrya can't convert in to json directly
            body['traffic_info']=result[0].tolist()
            # Now inser that data in to the SQL as a producer
            self.publisher.publish(body,routing_key=self.task_queue3.routing_key)
            print("Message Classified and Published to SQL queue")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            # Acknowledge, to remove the message from the queue
        finally:
            message.ack()
def runConsumer():
    data = settings()
    connection,queue = data.get_classifier_queue_setting()
    connection,exchange,_,_,task_queue3 = data.get_producer_settings()
    obj = TweetClassifier()
    publisher = Producer(connection,exchange=exchange)
    classifierConsumer(connection,queue,obj,publisher,task_queue3).run()
    consumer.close()
    connection.close()