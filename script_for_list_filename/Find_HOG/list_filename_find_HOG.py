import os

motorcycle_brand_name_to_write = "<file_name>";
motorcycle_brand_name= "<brandname>_roi\\"; #can be anything you want 
list_of_filename = os.listdir("./");
pathname = "\".\<data_set_folder>\\<brand_name>\\";
for eachname in list_of_filename :
        command = "finding_hog_features.exe ";
        command += pathname + motorcycle_brand_name + eachname;
        print(command, end = "\" " + motorcycle_brand_name_to_write + "\n");
