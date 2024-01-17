# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import os, shutil
import xlwt
from xlwt import Workbook
import xlrd
from PIL import Image

def copy_files(args):
    exel_source = args.output_dir + '\porduction_result.xls'
    dest_Fold = args.output_dir

    wb = xlrd.open_workbook(exel_source)

    for s in range(2):
        sheet = wb.sheet_by_index(s)
        if s == 0:
            dest = dest_Fold + '/ROI'
            if not os.path.isdir(dest):
                os.mkdir(dest)
        elif s == 1:
            dest = dest_Fold + '/Non-ROI'
            if not os.path.isdir(dest):
                os.mkdir(dest)
        for r in range(1, sheet.nrows):
            file = sheet.cell_value(r, 0)
            shutil.copy(file, dest)
