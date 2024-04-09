# Planned Data Analytics Examples 


Since the episode metadata is dataframe that is very easy to work with, we demonstrate
the capability with the following examples that work on the actual step data. 
* **extract and group columns**: we extract natural language instruction from steps and use it to tag episodes (done)
* **batch transformation**: we resize images. This involves creating a column, resizing images, adding a new column to store the images, and save the transformation
* **tagging** This runs yolo on the first frame and save the tag to the metadata
* **summary stats** aggregate a dataset-wise average of a matrix 