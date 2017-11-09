import sys
filenames = sys.argv[1:];
with open('./concatenated_file.txt', 'w') as outputfile:
    for each_filename in filenames:
        with open(each_filename, 'r') as inputfile:
            for line in inputfile:
                outputfile.write(line);

outputfile.close();
inputfile.close();
