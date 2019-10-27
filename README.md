# UCSF_QBI_Hackathon_ImageSegmentor
The Image Segmentation scripts for Josh and Noel at Optical Biosystems. 

Josh had flourcent images with DAPI labeling the neuclei of neurons in mouse brain slices. 
The team used a modified watershed, Voronoi and U-NET CNN to do the segmentations. In addition to segmenting the neuclei the neuclei centers were found using centroids method.

--------------------------------------------------------------------------------------------------------------------------
Watershed: https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

--------------------------------------------------------------------------------------------------------------------------
Voronoi: 

  Methods:
  -thresholded grayscale image of cells using otsu
  -found islands (cells) and averaged all pixel locations to find centroids
  -performed voronoi using scipy on centroids
  
 -------------------------------------------------------------------------------------------------------------------------
 
U-NET CNN:https://arxiv.org/abs/1505.04597
