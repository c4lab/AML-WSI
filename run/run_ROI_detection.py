import os
output_path = "path/to/your/output/directory/"
patches_path = "path/to/your/patches/directory/"
main_path = "../ROI_detection/"
exist_slide_list = os.listdir(output_path)
slide_list =[]
for i in os.listdir(patches_path):
    if not i in exist_slide_list:
        slide_list.append(i)
slide_list.sort(
    key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
)
for i in slide_list:
  input_path = f"{patches_path}{i}/rightside_patch"
  if os.path.exists(input_path):
    os.mkdir(f"{output_path}{i}")
    os.system(f"python {main_path}main.py --predict-mode --report-excel --data-path {input_path} --threshold 0.8 --output-dir {output_path}{i} --down-scale 1 --batch-size 32")
  else:
    print(f"{input_path} is not exist")