from mysql.connector import connect, Error
# DB Configuration
class mysqlInsert:
    def __init__(self):
        self.config = {'user': 'saukumar',
                       'password': '12345678',
                       'host': 'test.cj0nucuodepu.us-east-2.rds.amazonaws.com',
                       'database': 'tweet',
                       'raise_on_warnings': True,
                       }
        self.insert_query =("INSERT INTO tweet_data "
                            "(tweet_id, tweet_text,tweet_date,user_id,user_loc,twitter_geo_loc) "
                            "VALUES (%s,%s,%s,%s,%s,%s)")
        self.update_query_classifier = (""" UPDATE tweet_data
                                        SET traffic_info = %s
                                        WHERE tweet_id = %s """)
        self.cnx = connect(**self.config)
        self.cursor = self.cnx.cursor()
        
    def dbConnect(self):
        # when need to resetablish the connection
        self.cnx = connect(**self.config)
        self.cursor = self.cnx.cursor()
    def insertData(self,tweet):
        # db connection check
        
        if not self.cnx.is_connected():
            self.dbConnect()
        try:
            # Insert new data
            # below three lines for the emoji
            self.cursor.execute('SET NAMES utf8mb4')
            self.cursor.execute("SET CHARACTER SET utf8mb4")
            self.cursor.execute("SET character_set_connection=utf8mb4")
            
            tweet_text = tweet['text']
            tweet_id = tweet['tweet_id']
            tweet_date =  tweet['date']
            user_id = tweet['user_id']
            user_loc = tweet['user_loc']
            
            # check whether location is null or not
            if tweet["tweet_loc"]:
                # it will be like [41.8781136, -87.6297982]
                twitter_geo_loc = str(tweet["tweet_loc"]['coordinates'])
            else:
                twitter_geo_loc = tweet['tweet_loc']
                
            tweet_data = (tweet_id,tweet_text,tweet_date,user_id,user_loc,twitter_geo_loc)
            self.cursor.execute(self.insert_query, tweet_data)
            # Make sure data is committed to the database
            self.cnx.commit()
            print("Record Inserted")
        except Error as error:
            print("Record Insert fail, might be same data, check more detail at SQL insert")
            #print(tweet_data)
            #print(error)
        finally:
            pass
            # do not close connection
            #self.cursor.close()
            #self.cnx.close()
    def insertclassifrInfo(self,data):
        if not self.cnx.is_connected():
            self.dbConnect()
        tweet_data = (data['traffic_info'],data['tweet_id'])
        try:
            self.cursor.execute(self.update_query_classifier, tweet_data)
            # Make sure data is committed to the database
            self.cnx.commit()
            print("Record Updated")
        except Error as error:
            print("Record Update Failed, check more detail at SQL Inser")
            #print(error)
        finally:
            pass
            # do not close connection
            #self.cursor.close()
            #self.cnx.close()        
    def inserLocationInfo(self,data):
        pass
    def insert(self,data):
        # function that will going to decide whether to update the classifier 
        # or geo location or insert the tweet information 
        if data['data_type']=="tweet_info":
            self.insertData(data)
        elif data['data_type'] == "classifier_info":
            self.insertclassifrInfo(data)
        else:
            self.inserLocationInfo(data)
if __name__ == '__main__':
    obj = mysqlInsert()
    tweet = {'date': '2014-06-06 21:38:44',
     'text': 'The #SurreyTMC wishes everyone a wonderful holiday season. Remember to plan for transportation ahead of time and do\xe2\x80\xa6 https://t.co/8pxsqNybQM',
     'tweet_id': 944344722167287908L,
     'tweet_loc': {'coordinates': [41.8781136, -87.6297982], 'type': 'Point'},
     'user_id': u'SurreyTraffic',
     'user_loc': 'Surrey, BC, Canada'}
    tweet['data_type'] = 'tweet_info'
    obj.insert(tweet)
    tweet['traffic_info'] = 0
    tweet['data_type'] = 'classifier_info'
    obj.insert(tweet)