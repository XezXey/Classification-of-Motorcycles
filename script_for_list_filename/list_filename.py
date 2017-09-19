import os

motorcycle_brand_name = "<brand_name>\\";	#can be anything you want
list_of_filename = os.listdir("./");
pathname = "\".\data_set_phase1\\";
for eachname in list_of_filename :
        command = "extract_features_from_images.exe 1 ";
        command += pathname + motorcycle_brand_name + eachname;
        print(command, end = "\"\n");
