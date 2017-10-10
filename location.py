from requests import get
USER_KEY = 'Avj91lXJ62VAoqK7fvlBZSFtzXZlmrIyblEg6tiJpInGv8llcTxRrez7gUmxO5wx'
location = 'west michigan street'
location_query = "%".join(location.split())
url_ = 'http://dev.virtualearth.net/REST/v1/Locations?q='+ location_query +'&key=' + USER_KEY
response = get(url = url_)
data = response.json()
geo_coordinate = data['resourceSets'][0]['resources'][0]['point']['coordinates']