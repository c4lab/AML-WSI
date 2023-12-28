# MIL_gene_prediction_WSI_AML
multiple instance learning for predicting gene mutations from whole slide images of acute myeloid leukemia

![The idea](./figs/Final_method.png) 

<p align="center">
 Overview of the proposed method.</center>
</p>

 ## Using PyHIST for Patches Generation

 This process and code is based on [PyHIST](https://github.com/manuel-munoz-aguirre/PyHIST).

For single WSI:
    
    python pyhist.py --content-threshold 0.05 --output /path/to/your/output/directory --output-downsample 1 --save-patches --save-tilecrossed-image --info "verbose" /path/to/your/WSI

For a WSI directory set:

    python /run/run_pyHIST.py

## ROI Detection

The model architecture and code is based on [DenseNet121](https://doi.org/10.5281/zenodo.6373429).

For all patches in a single WSI:
    
    python /ROI_detection/main.py --predict-mode --report-excel /path/to/your/patches/directory/ --threshold 0.8 --output-dir /path/to/your/output/directory --down-scale 1 --batch-size 32

For a WSI directory set:

    python /run/run_ROI_detection.py

## Cell Detection
## System Requirements
- Python 3.9.16
- yolov4 2.0.3


