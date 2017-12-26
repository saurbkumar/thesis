from pymongo import MongoClient
class mongoDataInsert():
    def __init__(self):
        self.connect = 'mongodb://saukumar:1234@ds159926.mlab.com:59926/tweets'
        self.client = MongoClient(self.connect)
        self.db = self.client.tweets# tweets --> database name
        self.tweet_table = self.db.data# data is our table
    def insert(self,data):
        try:
            self.tweet_table.insert_one({'_id':data['tweet_id'],'tweet_body':data})
            #print("Data Inserted")
        except:
            print('Data not inserted either due to duplicate key or something else')