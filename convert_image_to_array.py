# Import the necessary libraries
from PIL import Image
import numpy as np


def ndarray2GSLIB(array, data_file, col_name):
    """
    Converts the array with the image data into a ti.dat file,
    which will be used as a training image. This file must be in the format as in: 
    https://mpslib.readthedocs.io/en/latest/training-image-format.html

    Parameters:
        array: array_like(int) 
            array obtained from training image
        data_file: str
            file name
        col_name: str 
            column name
    """    
    ny = (array.shape[0])
    nx = (array.shape[1])
    ncol = 1
    file_out = open(data_file, "w")
    file_out.write(str(nx) + ' ' + str(ny)  + ' ' + str(ncol) + '\n')  
    file_out.write(str(ncol) + '\n')  
    file_out.write(col_name  + '\n') 
    if array.ndim == 2:
        for iy in array:
            for ix in iy:
                file_out.write(str(ix)+ '\n')                  
    else:       
        print("Error: must use a 2D array")            
        file_out.close()
        return            
    file_out.close()
    return 


def convertImageToGSLIBFile(image):
    img =  Image.open(image)
    numpyData = np.asarray(img)
    ndarray2GSLIB(numpyData, 'ti.dat', 'channel1')
    return numpyData


def convertImageToHardData(image):
    img =  Image.open(image)
    numpyData = np.asarray(img)
    ny = (numpyData.shape[0])
    nx = (numpyData.shape[1])
    nz = 0
    hardData = []
    for index_x, ix in enumerate(numpyData):
        for index_y, iy in enumerate(ix):
            hardData.append([index_y*(33.61*10**(-6)), index_x*33.61*10**(-6), nz, iy])
    return np.array(hardData)



