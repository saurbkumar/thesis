import tweepy
class tweetDataExtraction:
    def __init__(self):
        # For twitter api
        self.consumer_key = "Hol90i780joDoqDzWS32tR2cn"
        self.consumer_secret = "wIPsoeGyHbqfmHNcCdATs8GOlPOx9HeU5OlekcGm6D2TtHyUPk"
        self.access_token = "249152008-zYoxFHAVeDzlWNasuaqxOXBOZpihHCYxi0frmChO"
        self.access_token_secret = "qBTozbbXA10mdI57sEhOoiYrIE18E2GHg8qCwKnkjNZYl"
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.twitter_api = tweepy.API(self.auth)
        
        # twitter ids
        self.twitter_id = ["@SarahJindra","@wazetrafficchi","@dtptraffic",
              "@TrafflineDEL","@TrafflineCHN","@TrafflineIndore",
              "@TotalTrafficCHI","@TfL","@HighwaysSEAST","@BillWest5",
              "@John_Kass","@iKartikRao","@shelvieana_prat","@prats_09",
              "@94_294_tollway","@DildineMedia","@CrazyRicardo",
              "@TotalTrafficNYC","@WazeTrafficNYC","@Traffic4NY",
              "@NYTrafficAlert","@NYC_DOT","@511NYC","@NYCityAlerts",
              "@TotalTrafficNYC",'@SurreyTraffic']
        
        # key word list for data extraction using geolocation
        
        self.words = ['happy','rain','water','drain','traffic','road','block',
         'road accidents','accidents','congestion','construction']

        #city list
        self.chicago = "41.881832,-87.627760,30km"
        self.chennai = "13.083162,80.282758,30km"
        self.new_york = "40.714264,-73.978499,30km"
        self.london = "51.505234,-0.111244,30km"
        self.newdelhi = "28.612952,77.211953,30km"
        self.indianapolis = "39.767927,-86.158749,30km"
        self.bombay ="19.110914,72.885140,30km"
        self.new_jersey = "40.279865,-74.517549,30km"
        
        self.geo_locations = [self.chicago,self.new_york,self.london,self.indianapolis]

    def getData(self):
            def userIdDataExtr(twitter_id,twitter_api):
                tweets_ = []
                all_tweetInfo = []
                # Extract tweets wrt to user account
                for user in twitter_id:
                    #print(user)
                    for tweet in twitter_api.user_timeline(id=user,count = 1):
                        tweet_date =  tweet.author.created_at.strftime('%Y-%m-%d %H:%M:%S')
                        user_id = tweet.author.screen_name
                        location = tweet.author.location.encode('utf-8')
                        geoLoc = tweet.geo
                        data = {'tweet_id':tweet.id,'text':tweet.text.encode('utf-8'),'date':tweet_date,'user_id':user_id,'user_loc':location,'tweet_loc':geoLoc}
                        tweets_.append(data)
                        all_tweetInfo.append(tweet)
                    #end of loop
                return tweets_
                
            def geoDataExtr(geo_locations,words,twitter_api):
                # Extract tweets wrt to the city location and keywords
                tweets_ = []
                all_tweetInfo = []
                for city in geo_locations:
                    for word in words:
                        for tweet in twitter_api.search(q=word, geocode=city,count=1):
                            tweet_date =  tweet.author.created_at.strftime('%Y-%m-%d %H:%M:%S')
                            user_id = tweet.author.screen_name
                            location = tweet.author.location.encode('utf-8')
                            geoLoc = tweet.geo
                            all_tweetInfo.append(tweet)
                            data = {'tweet_id':tweet.id,'text':tweet.text.encode('utf-8'),'date':tweet_date,'user_id':user_id,'user_loc':location,'tweet_loc':geoLoc}
                            tweets_.append(data)
                return tweets_
            # here need to insert logic for the all_tweet_info
            allTweets = []
            allTweets.extend(geoDataExtr(self.geo_locations,self.words,self.twitter_api))
            allTweets.extend(userIdDataExtr(self.twitter_id,self.twitter_api))
            return allTweets
if __name__ == "__main__":
    obj = tweetDataExtraction()
    tweets = obj.getData()