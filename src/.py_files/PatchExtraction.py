#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import SimpleITK as sitk
import pathlib as pathlib
import numpy as np 
import matplotlib.pyplot as plt

'''
Author: Sal Choueib
Student No. : 10188460
Class: CISC881
Description:

    This script will read in two .mhd volumes, the first is the full abdominal CT, the second is the segmented cyst volume.
    It will also read in the corresponding clinical stratifications from an excel file. 
    First it will split the data set into a train and independent test set to later be used in model testing. The split it 
    applied via sci kit learns' train test split function, and is implemented on just the abdominal CT volume. 
    The code will keep track of the patient ID so that the same patients can be extracted from the segmented cyst volumes. 
    Once the split is finished, the script removes the ID's and will run the data through the patch extraction functin
    
    
    in this function, the abdominal CT and the corresponding segmented cyst CT will be read in, and the latter volume
    will be used to find the center of mass for the patch extraction. This will be at the center of mass of the cyst 
    of each slice. Therefore, only slices containing cysts will be extracted. During the patch extraction, intensity
    values of the patches are normalized. he slices are then stacked in a numpy stack and written to disk. 

'''

def normalize(patch):
    max_val = patch.max()
    ct_arr = patch / max_val
    return ct_arr


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
        

def patch_extraction(img_list, seg_list, patch_size,labels,patient_list):
    #patient_list is simply for console output clarity. 
    
    if len(img_list) != len(seg_list):
        print("Image list and segmented list did not correspond: \n")
        print("Length of image list: " + str(len(img_list)) + "\nLength of segmentation list: " +str(len(seg_list)))
        return
    elif patch_size%2 != 0:
        print("Patch size must be even.\nPatch size: " +str(patch_size))
        return
    
    
    label_indx    = 0
    for patient_index in range(len(img_list)):

        #select co-registered volumes for patient x
        
        img_arr = img_list[patient_index]
        seg_arr = seg_list[patient_index]

        slice_list = []
        #create an index of slices that contain the segmented cyst
        #to use for the slice extraction
        for i in range(len(seg_arr)):
            ct_slice = seg_arr[i,:,:]
            max_val  = ct_slice.max()
            if max_val != -1000:
                slice_list.append(i)

        #Use the slice index to extract patches 
        for slice_indx in range(len(slice_list)-1):
            seg_slice = seg_arr[slice_list[slice_indx],:,:]
            img_slice = img_arr[slice_list[slice_indx],:,:]

            row_sum = 0
            col_sum = 0
            count   = 0
            
            # calculate the center of mass of the cyst from the segmented cyst volume
            for row in range(seg_slice.shape[0]):
                for col in range(seg_slice.shape[1]):
                    if seg_slice[row][col] != -1000:
                        row_sum += row
                        col_sum += col
                        count   += 1

            row_center = round(row_sum/count)
            col_center = round(col_sum/count)
            
            print("Patient: " + str(patient_list[patient_index]))
            print("Patient index: " +str(patient_index))
            print("Patient "+str(patient_index)+ ", slice "+str(slice_indx)+":")
            print("Tumor center of mass: ["+str(row_center)+","+str(col_center)+"]")

            row_min = int(row_center-(patch_size/2))
            row_max = int(row_center+(patch_size/2))

            col_min = int(col_center-(patch_size/2))
            col_max = int(col_center+(patch_size/2))
            
            #from the center of mass of the cyst take all the pixels from range 0 to patch_size/2 in all directions. 
            patch   = img_slice[row_min:row_max,col_min:col_max]
            
            if patient_index == 0 and label_indx==0:
                global ct_patches
                global ct_patch_labels
                ct_patches      =  normalize(patch)
                #ct_patches      =  patch
                ct_patch_labels = labels[patient_index]
                label_indx+=1
                print(ct_patch_labels)
                print("Patch Label to labels: " + str(labels[patient_index]))
            else:
                ct_patches      = np.dstack((ct_patches,normalize(patch)))
                ct_patch_labels = np.vstack((ct_patch_labels,labels[patient_index]))
                print("Patch Label to labels: " + str(labels[patient_index]))
            
            print('-------------------------')
    return ct_patches,ct_patch_labels


# In[20]:


#Will extract the patient id's from the already split abdominal CTs, and use that 
#to split the segmented CTs exactly. 
def split_data(ct_train,ct_test,seg_cyst_images):
    cyst_train=[]
    cyst_test =[]
    temparr=seg_cyst_images #to be able to remove elements from list as chosen, for comp. efficiency.
    
    for elem in ct_train:
        pid = elem[0]
        arr = elem[1]
        for seg in temparr:
            if seg[0] == pid:
                cyst_train.append(seg)
                temparr.remove(seg)

    for elem in ct_test:
        pid = elem[0]
        arr = elem[1]
        for seg in seg_cyst_images:
            if seg[0] == pid:
                cyst_test.append(seg)
                seg_cyst_images.remove(seg)
        
    return cyst_train,cyst_test


def remove_id(tuple_list):
    no_id_list=[]
    for patient in tuple_list:
        no_id_list.append(patient[1])
    return no_id_list


# In[21]:


import os
import SimpleITK as sitk
import pathlib as pathlib
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#initializations and input paths
patch_size             =  80 #must be even
input_dir_whole_image  = "C:/Users/salch/pancreasCNN/data/resampled_whole_image"
input_dir_seg_panc     = "C:/Users/salch/pancreasCNN/data/resampled_seg_cyst_nn"
input_dir_clinicalData = "C:/Users/salch/Desktop/THESIS/clinicalData.xlsx"

output_dir_ct_patches_train  = "C:/Users/salch/pancreasCNN/data/patch_extraction/train/"
output_dir_ct_patches_test   = "C:/Users/salch/pancreasCNN/data/patch_extraction/test/"


# Loop through the directory containing the CT volumes and convert them to numpy arrays
# and append them to a list. 
def data_access(input_path): 
    # will create a list of numpy arrays representing volumes from input_path
    
    input_Data   = pathlib.Path(input_path)# path to resampled .mhd files 
    patientFiles = [file for file in input_Data.iterdir()] 
    
    image_list   = []
    for volume in patientFiles: 
        if ".mhd" in str(volume):
            pid = str(volume.stem[0:4])
            img     = sitk.ReadImage(str(volume),imageIO = "MetaImageIO")
            img_arr = sitk.GetArrayFromImage(img)
            image_list.append((pid,img_arr))
    
    return image_list
#---------------------------------------------------------------------------------

# Will read in excel file containing clinical data and convert it to panda datafram
# 
def import_excel(path_clinData):
   
    df               = pd.read_excel(path_clinData)
    df               = df[['ipmn_id', 'risk_strat']]
    df['risk_strat'] = df['risk_strat'].map({'High risk': 1, 'Low risk': 0}) #change label from str to binary value
    df               = df.sort_values(by=['ipmn_id']) #sort the list based on 
    
    return df
#---------------------------------------------------------------------------------

#Then will extract just the label data and sort based 
def label_extraction(input_to_clinData):
    
    labels       = import_excel(input_dir_clinicalData)
    patient_list = labels["ipmn_id"] #not for funtional purpose, just for output
    patient_list = np.expand_dims(patient_list,axis=1)
    labels       = labels["risk_strat"]
    labels       = np.expand_dims(labels,axis=1)
    
    return labels, patient_list
#---------------------------------------------------------------------------------
def save_patches_to_disk(ct_patches,ct_patch_labels,output_dir_ct_labels):
    
    np.save(output_dir_ct_labels+"ct_patches.npy",ct_patches)
    np.save(output_dir_ct_labels+"labels.npy",ct_patch_labels)
    return



def main():

    
    whole_images                              = data_access(input_dir_whole_image) #returns array of CT volume
    seg_cyst_images                           = data_access(input_dir_seg_panc)
    labels,patient_list                       = label_extraction(input_dir_clinicalData)
    
    make_directory(output_dir_ct_patches_train)
    make_directory(output_dir_ct_patches_test)


    #split the data
    
    seed = 7
    numpy.random.seed(seed)
    ct_train, ct_test, labels_train, labels_test = train_test_split(whole_images, labels, test_size=0.25,stratify=labels, random_state=seed)
    
    cyst_train,cyst_test = split_data(ct_train,ct_test,seg_cyst_images)
    
    cyst_train = remove_id(cyst_train)
    cyst_test  = remove_id(cyst_test)
    ct_train   = remove_id(ct_train)
    ct_test    = remove_id(ct_test)
    
    #extraxt the patches
    ct_patches_train, ct_patch_labels_train = patch_extraction(ct_train,
                                                                 cyst_train,
                                                                 patch_size,
                                                                 labels_train,
                                                                 patient_list)
    
    ct_patches_test, ct_patch_labels_test = patch_extraction(ct_test,
                                                                 cyst_test,
                                                                 patch_size,
                                                                 labels_train,
                                                                 patient_list)
    # write ptaches to disk
    save_patches_to_disk(
        ct_patches_train,
        ct_patch_labels_train,
        output_dir_ct_patches_train,
    )

    
    save_patches_to_disk(
        ct_patches_test,
        ct_patch_labels_test,
        output_dir_ct_patches_test,
    )
    
    check()
    
main()


# In[24]:


def check():
    print("Patch shape (img_dim, img_dim, img_number): ")
    print(ct_patches_train.shape)
    print(ct_patch_labels_train.shape)
    print("\n\nTraining set positive and negative observation balance: ")
    print("Negative observations (low risk)  : " + str(np.count_nonzero(ct_patch_labels_train==0)))
    print("Postive observations  (high risk) : " + str(ct_patches_train.shape[2]-(np.count_nonzero(ct_patch_labels_train==0))))

    print("\n\nTest set positive and negative observation balance: ")
    print("Negative observations (low risk)  : " + str(np.count_nonzero(ct_patch_labels_test==0)))
    print("Postive observations  (high risk) : " + str(ct_patches_test.shape[2]-(np.count_nonzero(ct_patch_labels_test==0))))

    print('\n\n')
    slics = ct_patches[:,:,156]
    print("Image Dimensions: " + str(slics.shape))
    plt.imshow(slics, cmap='gray')

    print("Label: " + str(ct_patch_labels[156]))
    
    
    #Check the dimensions are correct for the written files

    train = np.load("c:/users/salch/pancreasCNN/data/patch_extraction/train/ct_patches.npy")
    ltrain =np.load("c:/users/salch/pancreasCNN/data/patch_extraction/train/labels.npy")
    print(train.shape)
    print(ltrain.shape)


# In[25]:





# In[14]:





# In[ ]:




