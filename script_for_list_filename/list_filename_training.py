import os

motorcycle_brand_name = "<brand_name>\\<brand_name>_roi\\";	#can be anything you want
list_of_filename = os.listdir("./");
pathname = "\".\data_set_phase1\\";
for eachname in list_of_filename :
        command = "training_model ";
        command += pathname + motorcycle_brand_name + eachname;
        print(command, end = "\"\n");
