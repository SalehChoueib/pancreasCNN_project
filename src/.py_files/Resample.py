#!/usr/bin/env python
# coding: utf-8

# In[1]:



'''
Author: Sal Choueib
Student No. : 10188460
Class: CISC881
Description:

    This script will read in the raw CT volumes as .mhd files using sitk and will resample the images
    based on the most common voxel spacing found in the raw data. The image dimensions will also 
    be resampled based on the original dimensions and the new voxel spacing. 
    
    For the segmented cyst volumes we use a nearest neighbor interporlator. 

'''


import os
import SimpleITK as sitk
import pathlib as pathlib

voxel_spacing = (0.82,0.82,2.5) #most common voxel spacing. 
#voxel_spacing = (1,1,1)




def make_directory(path):
    try:
        os.mkdir(path)
        
    except FileExistsError:
        print("-----FAILED: "+path+ " Already exists")
    except OSError:
        print("Failed to create directory in: " +path)
    else:
        print("Directory created in: " +path)
        return None
    try:
        os.makedirs(path)
    except FileExistsError:
        print("-----FAILED: "+path+ " Already exists")
    except OSError:
        print("Failed to create directory in: " +path)
    else:
        print("Directory created in: " +path)
        return None
        
def resample(image,seg):
    global voxelSpacing
    
    
    orig_spacing = image.GetSpacing()
    new_spacing  = voxel_spacing
    
    spacing_x = new_spacing[0]
    spacing_y = new_spacing[1]
    spacing_z = new_spacing[2]

    size = image.GetSize()

    fact_x = orig_spacing[0] / spacing_x
    fact_y = orig_spacing[1] / spacing_y
    fact_z = orig_spacing[2] / spacing_z

    size_x = int(round(size[0] * fact_x))
    size_y = int(round(size[1] * fact_y))
    size_z = int(round(size[2] * fact_z))
    
    
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(image)
    f.SetOutputOrigin(image.GetOrigin())
    f.SetOutputSpacing((spacing_x, spacing_y, spacing_z))
    f.SetSize((size_x, size_y, size_z))
    if seg ==1: 
        f.SetInterpolator(sitk.sitkLinear)
    if seg == 2:
        f.SetInterpolator(sitk.sitkNearestNeighbor)
    
    #f.SetOutputPixelType(sitk.sitkInt16)
    result = f.Execute(image)
    return result


def data_access(path,output_dir,seg):
    prostateData = pathlib.Path(path)
    patientFiles = [file for file in prostateData.iterdir()]
    for file in patientFiles:
        if '.mhd' in str(file):
            patientID = str(file.stem)
            writefile = output_dir+"/"+patientID+".mhd"
            reader = sitk.ReadImage(str(file), imageIO = "MetaImageIO")
            result= resample(reader,seg)
            sitk.WriteImage(result,writefile)



def main():
    
    #intializations & paths
    output_dir_whole_image = "C:/Users/salch/pancreasCNN/data/resampled_whole_image"
    output_dir_seg_panc = "C:/Users/salch/pancreasCNN/data/resampled_seg_cyst_nn"
    
    inputPath_whole_image="C:/Users/salch/Desktop/THESIS/pancreas_data/IPMN/data_whole_image"
    inputPath_seg_panc="C:/Users/salch/Desktop/THESIS/pancreas_data/IPMN/data_segmented_cyst"
    
    make_directory(output_dir_whole_image)
    make_directory(output_dir_seg_panc)
    
    #access, modify and write the new data. 
    data_access(inputPath_whole_image,output_dir_whole_image,1)
    data_access(inputPath_seg_panc,output_dir_seg_panc,2)
    
    #load new data to check if the resampling produces the intended results 
    print("The results of the resampled pancreatic data: \n\n")
    print("Patient 1261:\n")
    reader = sitk.ReadImage('C:/Users/salch/pancreasCNN/data/resampled_whole_image/1261_ipmn_preop_volume.mhd')
    print(reader.GetSpacing())
    print(reader.GetSize())
    
    print("\n\nPatient 1427:\n")
    reader = sitk.ReadImage('C:/Users/salch/pancreasCNN/data/resampled_whole_image/1427_ipmn_preop_volume.mhd')
    print(reader.GetSpacing())
    print(reader.GetSize())

main()
                
                        

