# Segmantation_based_Semantic_Matting [WIP]

This repository is unofficial implementation of paper **Instance Segmentation based Semantic Matting for Compositing Applications**(https://arxiv.org/abs/1904.05457)

Final Results of segmenatation are great, but not as good as the results proposed in the paper.

![Results](https://github.com/Griffin98/Automatic-Background-Removal/raw/master/Results.png)

### ToDo:
- OOPs Class Design
- Improve Trimap Generation
  * Trimap Generation Stage(Feedback Loop) seems to have major impact on the final output. My current implementation is sort of lazy implemenatation by just iterating kernel size in decreasing order. Need to work on it, in order to achieve accuracy as proposed in paper.
- Multiple Object Segmentation
  * Currently i have limited the code to segment only few instances from the COCO dataset.


### Run
1. Download Pre-trained Mask R-CNN Model from [Link](https://github.com/matterport/Mask_RCNN/releases)
2. Download Pre-trained Deep Image Matting model from [Link](https://github.com/foamliu/Deep-Image-Matting-v2/releases)
3. Place both the downloaded models in __models/__ directory

To Run:
> python demo_end_to_end.py <input_image>

Output:
> Data/foreground/ and Data/alpha
