import os
import xlrd
output_path = "path/to/your/output/directory/"
patches_path = "path/to/your/patches/directory/"
main_script_path = "../ROI_detection/"
excel_name="porduction_result.xls"
exist_slide_list = os.listdir(output_path)
slide_list =[]
for i in os.listdir(patches_path):
    if not i in exist_slide_list:
        slide_list.append(i)
slide_list.sort(
    key=lambda x: int(x[1:]) if x[1:].isdigit() else 0
)
for i in slide_list:
  excel_path = f"{patches_path}{i}/{excel_name}"
  if os.path.exists(excel_path):
    print(f"Processing {excel_path}")
    wb = xlrd.open_workbook(excel_path)
    sheet1 = wb.sheet_by_index(0)
    pos_list = sheet1.col_values(0)
    pos_list.pop(0)
    # sheet2 = wb.sheet_by_index(1)
    # neg_list = sheet2.col_values(0)
    # neg_list.pop(0)
    for patch in pos_list:
      os.system(f"python {main_script_path} --input_patch {patch} --output_dir {output_path} --cell_size 64")
    print(f"Finish {excel_path}")
  else:
    print(f"{excel_path} is not exist")