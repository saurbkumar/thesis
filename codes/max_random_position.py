"""
You need to write a function which will evenly return indexes of a max value in
 the array. In the example below max value is 6, and its positions are 3, 6 and 7.
 So each run function should return random index from the set. 
"""

data = [1,-2,0,6,2,-4,6,6]
max_ = data[0]
for number in data:
    if max_ < number:
        max_ = number
index = []
for number in data:
    if number == max_:
        index.append()