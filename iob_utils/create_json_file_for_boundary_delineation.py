"""
1- iterate over files in the train directory
2- read excel file
3- check which file corresponds to which line of the excel file
4- grab the information about x bounderies
5- generate a dictionary for each file
6- write the dictionary to the json file
"""
import json, csv
import os, glob


csv_filename = '/projects/parisa/data/test_detectron_oct/x_boxes.csv'
json_filename = '/projects/parisa/data/test_detectron_oct/train/x_boxes.json'
data_dir = '/projects/parisa/data/test_detectron_oct/train/'

'''
read csv file and store it in an array -> rows
check if the memory capacity if enough for storing the array 
'''
#fields=[]
rows=[]
c=0
with open(csv_filename, 'r') as csv_f:
    csv_reader = csv.reader(csv_f)
    fields = next(csv_reader)
    c=+1
    for row in csv_reader:
        if c<=246:
            rows.append(row)
            c+=1
print(c)

'''
iterate over png files and make a dictionary for each
then dump them into the json outfile
'''
with open(json_filename, 'w') as outfile:
    outfile.write("[")
    num_of_png_files = len(glob.glob(data_dir+'*.png'))
    for i,f in enumerate(glob.glob(data_dir+'*.png')):
        vol_name = os.path.basename(f)[:-12]+".vol"
        slice_idx = int(f[-7:][:-4])
        print("# processing:", os.path.basename(f))

        for row in rows:
            if vol_name == row[0] and slice_idx == int(row[2]):
                # make a dict for each png including filename and x_box
                row_idx = 5
                boxes = {}
                while not row[row_idx]=='':
                    x_box = [int(row[row_idx]), int(row[row_idx+1])]
                    y_box = [int(row[row_idx+2]), int(row[row_idx+3])]
                    boxes[str((row_idx-5)//4)] = {"x_box": x_box, "y_box": y_box}
                    row_idx += 4
                slice_dict = {
                    "file_name": os.path.basename(f),
                    "boxes": boxes,
                    }
                json.dump(slice_dict, outfile)
                if i<num_of_png_files-1: # to avoid adding comma to the end of last element
                     outfile.write(",\n")
    outfile.write("]")

'''
test if the generated files is ok
'''

with open(json_filename) as json_file:
    data = json.load(json_file)


