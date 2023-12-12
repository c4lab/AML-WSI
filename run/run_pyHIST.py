import os
WSI_type=".ndpi"
output_path = 'path/to/your/output/directory/'
slide_path = 'path/to/your/WSIs/directory/'
pyhist_path='path/to/your/pyHIST/directory/'
slide_list = []
for file in os.listdir(slide_path):
    if file.endswith(WSI_type):
        file_num = file.split(".")[0]
        if not file_num in os.listdir(output_path):
            slide_list.append(slide_path+file)
print(len(slide_list))
for i in slide_list:
    os.system(f'python {pyhist_path}pyhist.py --content-threshold 0.05 --output {output_path} --output-downsample 1 --save-patches --save-tilecrossed-image --info "verbose" {i}')