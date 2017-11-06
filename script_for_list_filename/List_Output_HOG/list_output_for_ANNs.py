###
ANNs number of output is depend on your class
if you have 4 class to classified you need to have output for each class like this
class 1 : 1,0,0,0,
class 2 : 0,1,0,0,
class 3 : 0,0,1,0,
class 4 : 0,0,0,1,
So you can change this pattern in code at <ANNs_Output>
###

import os

motorcycle_count = 0;
motorcycle_brand_name= "<brand_name>\\"; #can be anything you want 
list_of_filename = os.listdir("./");
pathname = "\".\data_set_folder\\<brand_name\\";
for eachname in list_of_filename :
    if(eachname[-4:] == ".JPG") :
        print("<ANNs Output>", end = "\n");
        motorcycle_count += 1;
print(motorcycle_count, end = "\n");
