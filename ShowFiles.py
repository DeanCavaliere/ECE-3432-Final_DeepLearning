import os
import re

dummy = open('April20/DeepTempleCar/data/list/dummy.csv', 'a')

path = 'April20/DeepTempleCar/data/images'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    t = re.sub("April20/DeepTempleCar/", "", f)
    dummy.write(t+'\n')
    #print(re.sub("April20/DeepTempleCar/", " ", f))
dummy.close()