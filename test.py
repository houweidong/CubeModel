import os
label_path = '/home/new/dataset/new/labels1/jinshenyi.txt'
pic_path = '/home/new/dataset/new/pictures/jinshenyi'

picname_list = os.listdir(pic_path)
with open(label_path) as f:
    line = f.readline().strip()
    while line:
        line = line.split()
        pic_name = line[0]
        if line[0] not in picname_list:
            print(line)
        line = f.readline().strip()