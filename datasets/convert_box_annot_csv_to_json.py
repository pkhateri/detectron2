"""
1. read the csv file and store it in a list
2. iterate over the list and grab the information about xy bounderies
3. generate a dictionary for each file
4. write the dictionary to the json file
"""
import json, csv
import os, glob
import random

'''
input parameters
'''
csv_filename = '/projects/parisa/data/progstar/box_annot/box_annotation_final.csv'
json_train_filename = '/projects/parisa/data/progstar/box_annot/box_annotation_train.json'
json_val_filename = '/projects/parisa/data/progstar/box_annot/box_annotation_val.json'
n = 350 # number of lines in the excel file to be read
k = 80 # percentage of data for train


'''
read csv file and store it in an array -> rows
TODO: check if the memory capacity is enough for storing the array 
'''
fileds=[]
rows=[]
c=0
with open(csv_filename, 'r') as csv_f:
    csv_reader = csv.reader(csv_f)
    fields = next(csv_reader) # The line will get the first row of the csv file (Header row)
    c+=1
    for row in csv_reader:
        if c<n: #read the first n line only
            rows.append(row)
            c+=1
'''
randomly choose k% of data for train and (100-k)% for val
'''
num_train = round(k*n/100)
num_val = n - num_train
id_train = random.sample(range(n),num_train)
id_train.sort()


'''
iterate over rows of csv file and make a dictionary for each png
then dump them into the json outfiles
'''
outfile_train = open(json_train_filename, 'w')
outfile_val = open(json_val_filename, 'w')
outfile_train.write("[")
outfile_val.write("[")
for i,row in enumerate(rows):
    box_num = 0
    boxes = {}
    png_file = row[0]
    while (box_num+1)*4<len(row) and not row[(box_num+1)*4]=='': #the first x1,x2,y1,y2 starts from row[4] (accidentally). the next box is the next 4 columns and so on till there is void ''
        x_box = [float(row[(box_num+1)*4]), float(row[(box_num+1)*4+1])]
        y_box = [float(row[(box_num+1)*4+2]), float(row[(box_num+1)*4+3])]
        boxes[str(box_num)] = {"x_box": x_box, "y_box": y_box}
        box_num += 1
    slice_dict = {
                "file_name": png_file,
                "boxes": boxes,
                }
    # choose if train or val
    if i in id_train:
        json.dump(slice_dict, outfile_train)
        outfile_train.write(",\n")
    else:
        json.dump(slice_dict, outfile_val)
        outfile_val.write(",\n")

outfile_train.close()
outfile_val.close()

# the end of the file: remove extra comma and add closing bracket.
os.system("sed -i '$ s/.$/]/' '%s'"%(json_train_filename))
os.system("sed -i '$ s/.$/]/' '%s'"%(json_val_filename)) 


'''
test if the generated files is ok
'''
with open(json_train_filename) as json_file:
    data = json.load(json_file)
