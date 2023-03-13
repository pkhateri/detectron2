import os, glob
from random import shuffle
import getopt, sys

'''
example:
python shuffle_divide_copy_files.py -i ../data/heyex_export_raw/renamed_vol/y49/19023_OS_20160525_14h19m25s_y49_z496_x1024/ -o ../data/test_detectron_oct/
''' 

def main(argv):
    input_dir = ''
    output_dir = ''
    opts, args = getopt.getopt(argv,"hi:o:",["input_dir=","output_dir="])
    for opt, arg in opts:
        if opt == '-h':
            print ('shuffle_divide_copy_files.py -i <input_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", "--input_dir"):
           input_dir = arg
        elif opt in ("-o", "--output_dir"):
           output_dir = arg
    
    
    input_files = glob.glob(input_dir+"/*.png")
    shuffle(input_files)
    n = len(input_files)
    n_train = round (0.8*n)
    n_val = n - n_train
    
    for i,f in enumerate(input_files):
        if i<n_train:
            cmd='cp -p "%s" "%s"'%(f, output_dir+"/train/")
            #print(cmd) # copy files
            os.system(cmd)
        else:
            cmd='cp -p "%s" "%s"'%(f, output_dir+"/val/")
            #print(cmd) # copy files
            os.system(cmd)

if __name__ == "__main__":
   main(sys.argv[1:])
