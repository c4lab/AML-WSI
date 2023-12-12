import os
output_path = "/home/weber50432/AML_image_processing/HCT_region_select/output/"
slide_path = "/home/exon_storage1/aml_slide/patches/"
exist_slide_list = os.listdir(output_path)
slide_list =[]
for i in os.listdir(slide_path):
    if not i in exist_slide_list:
        slide_list.append(i)
slide_list.sort(
    key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
)
for i in slide_list:
  input_path = "{}{}/rightside_patch".format(slide_path,i)
  if os.path.exists(input_path):
    os.mkdir("{}{}".format(output_path,i))
    os.system("python main.py --predict-mode --report-excel --data-path {} --threshold 0.8 --output-dir {}{} --down-scale 1 --batch-size 32".format(input_path,output_path,i) )
  else:
    print("{} is not exist".format(input_path))