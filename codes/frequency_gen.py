a = [1,2,2,3,4,4,5,5,5]
curr_bird_num = 1
curr_bird_type = 1
counter = 1
for i in range(len(a)):
    if i > 0:
        if a[i]==a[i-1]:
            counter = counter + 1
        else:
            if curr_bird_num < counter:
                curr_bird_num = counter
                curr_bird_type = a[i-1]
            if curr_bird_num == counter:
                if curr_bird_type > a[i-1]:
                    curr_bird_type = a[i-1]
            
            counter = 1
    if i==(len(a)-1):
            if curr_bird_num < counter:
                curr_bird_num = counter
                curr_bird_type = a[i-1]
            if curr_bird_num == counter:
                if curr_bird_type > a[i-1]:
                    curr_bird_type = a[i-1]        