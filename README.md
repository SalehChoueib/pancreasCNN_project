# pancreasCNN
##CISC 881 Final Project

This repo consists of five source code files that make up the 
pipeline of work: Resample, PatchExtraction, MedianFiltering, Data Augmentation, and Model_build. 

The pipeline begins with the raw CT volume data in the form of '.mhd' files and provides an end-to-end solution model builder for the stratification of these volumes into high and low risk malignancy. 

The pipeline is illustrated in figures/currentpipeline.

The source code is saved as jupyter notebooks to show the output of each script, since the data is private and may not be released. 

To run this code, simply follow the pipeline and modify the input and output path variables. Note that the output path of one file will be the input path of the next. 
