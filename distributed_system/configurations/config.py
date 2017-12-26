from kombu.connection import Connection
from kombu import Queue,Exchange

# Basic settings for queue, routing key and connection
class settings():
    def __init__(self):
        self.connection = Connection('amqp://kgmcrbkn:JEWBpwC2KUnSahYN680d9lkpqr8eh62o@fish.rmq.cloudamqp.com/kgmcrbkn')
        # exchange - Direct connection 
        self.dataExtr_exchange = Exchange(name='dataExtractor',type='direct')
        # Two queuse, one is for Mongodb and second is for the classifier
        
        # Mongo Queue
        self.mongo_queue = Queue(name ='mongo_queue',exchange=self.dataExtr_exchange, routing_key='mongo')
        # classifier queue
        self.classifier_queue = Queue(name ='classifier_queue',exchange=self.dataExtr_exchange, routing_key='classifier')
        # MySQl queue
        self.sql_queue = Queue(name ='sql_queue',exchange=self.dataExtr_exchange, routing_key='sql')
    
    def get_producer_settings(self):
        return[self.connection,self.dataExtr_exchange,self.mongo_queue,self.classifier_queue,self.sql_queue]
    
    def get_mongo_queue_setting(self):
        return[self.connection,self.mongo_queue]
    
    def get_classifier_queue_setting(self):
        return[self.connection,self.classifier_queue]
    
    def get_sql_queue_setting(self):
        return[self.connection,self.sql_queue]


