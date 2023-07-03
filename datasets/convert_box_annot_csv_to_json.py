"""
1. read the csv file and store it in a list
2. iterate over the list and grab the information about xy bounderies
3. generate a dictionary for each file
4. write the dictionary to the json file
"""
import json, csv
import os, glob, sys, getopt
import random

def read_csv_file(csv_filename, n):
    '''
    1. read csv file and store it in an array -> rows
    TODO: check if the memory capacity is enough for storing the array
    '''
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
    return rows

def shuffle_divide_dataset(n_total, k):

    '''
    randomly choose k% of data for train and rest for val and test
    '''
    n_train = round(k*n_total/100)
    n_val = round((n_total - n_train)/2)
    n_test = n_total - n_train - n_val
    train_idxs = random.sample(range(n_total),num_train)
    train_idxs.sort()
    val_idxs = random.sample(list(set(range(n_total)) - set(train_idxs)), num_val)
    val_idxs.sort()
    test_idxs = list(set(range(n_total)) - set(train_idxs) - set(val_idxs))
    test_idxs.sort()
    return train_idxs, val_idxs, test_idxs

def write_json_files(rows, k, json_train_filename, json_val_filename, json_test_filename):
    '''
    iterate over rows of csv file and make a dictionary for each png
    then dump them into the json outfiles
    '''
    n_total = len(rows)
    train_idxs, val_idxs, test_idxs = shuffle_divide_dataset(n_total, k)
    outfile_train = open(json_train_filename, 'w')
    outfile_val = open(json_val_filename, 'w')
    outfile_test = open(json_test_filename, 'w')
    outfile_train.write("[")
    outfile_val.write("[")
    outfile_test.write("[")
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
        # choose if train or val or test
        if i in train_idxs:
            json.dump(slice_dict, outfile_train)
            outfile_train.write(",\n")
        elif i in val_idxs:
            json.dump(slice_dict, outfile_val)
            outfile_val.write(",\n")
        else:
            json.dump(slice_dict, outfile_val)
            outfile_val.write(",\n")

    outfile_train.close()
    outfile_val.close()
    outfile_test.close()

    # the end of the file: remove extra comma and add closing bracket.
    os.system("sed -i '$ s/.$/]/' '%s'"%(json_train_filename))
    os.system("sed -i '$ s/.$/]/' '%s'"%(json_val_filename))
    os.system("sed -i '$ s/.$/]/' '%s'"%(json_test_filename))

    '''
    test if the generated files is ok
    '''
    with open(json_train_filename) as json_file:
        data = json.load(json_file)


def main(argv):
    '''
    0. input parameters:
    '''
    csv_filename = ""
    json_prefix = ""

    n = 1000 # number of rows to be read from the csv file
    k = 80 # percentage of data for train set

    opts, args = getopt.getopt(argv,"hc:d:j:n:k:",["csv_file=","json_dir=","json_prefix=", "num_rows=", "train_percentage="])
    for opt, arg in opts:
        if opt == '-h':
            print ('python convert_box_annot_csv_to_json.py -c <csv_filename> -d <json_dir> -j <json_prefix> -n <number_of_rows> k <train_percentage>')
            sys.exit()
        elif opt in ("-c", "--csv_file"):
            csv_filename = arg
        elif opt in ("-d", "--json_dir"):
            json_dir = arg
        elif opt in ("-j", "--json_prefix"):
            json_dir = arg
        elif opt in ("-n", "--num_rows"):
            n = int(arg)
        elif opt in ("-k", "--train_percentage"):
            n = int(arg)

    if len(argv) < 8:
        sys.exit("ERROR: all or part of input arguments are not provided!\n",
                 "python convert_box_annot_csv_to_json.py -c <csv_filename> -d <json_dir> -j <json_prefix> -n <number_of_rows> k <train_percentage>")

    if not os.path.exists(csv_filename):
        sys.exit("ERROR: csv_filename {} does not exist!".format(csv_filename))
    if not os.path.exists(png_dir):
        sys.exit("ERROR: json_dir {} does not exist!".format(json_dir))


    rows = read_csv_file(csv_filename, n)

    '''
    write json files
    '''
    json_train_filename = os.path.join(json_dir, json_prefix+"_train.json")
    json_val_filename = os.path.join(json_dir, json_prefix+"_val.json")
    json_test_filename = os.path.join(json_dir, json_prefix+"_test.json")
    write_json_files(rows, k, json_train_filename, json_val_filename, json_test_filename)
