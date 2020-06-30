# Setup
1. Prepare python environment, download dataset and pretrained 
faster rcnn model on visual genome according to README_neural_motifs.md.
Modify `VG_IMAGES` in `config.py` and `config_args.py` 
(the former is for detection, the latter is for generation).

2. If you are not run with V100 GPU, you may need to modify arch, code in lib/fpn/roi_align/src/cuda/Makefile, 
lib/fpn/roi_align/src/cuda/Makefile, lib/lstm/highway_lstm_cuda/make.sh
to match you GPU before make.

3. run `python combine_sg2im_neural_motifs/bbox_feature_dataset/extract_bbox_feature_each.py`
to do object detection and save the results for generation.

# Train and Test
For each experiment in 
https://docs.google.com/document/d/1ld3ZTIBhpFixb4ATW1UNvvlPTBBcDdkKXYppqEMV9LM/edit?usp=sharing,
you can search the order of it for training in one of `order` `order_2` ... with its `name`.

The order for testing can be found in one of `test_orders` `test_order2` ...

 