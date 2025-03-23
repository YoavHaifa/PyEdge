# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:09:56 2019
Handle real images
@author: yoavb
"""

import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

from MyString import GetValue, SetValue, RemoveValue, SetType, AddDesc


class CImage:
    """
    Class for holding an image
    Image is held as 32-bit float
    """
    def __init__(self, nLines=0, nCols=0, pData=None, bInit=False, fName=None, i = 0):
        """
        Args:
            imagesFileName: name of file to read - values are short - 512*512*nImages
        """
        matrix = GetValue(fName,'_matrix')
        if matrix:
            nLines = matrix
            nCols = matrix
        if nLines:
            self.nLines = nLines
        else:
            self.nLines = GetValue(fName,'_height')
        if nCols:
            self.nCols = nCols
        else:
            self.nCols = GetValue(fName,'_width')
        self.pData = pData
        self.fName = fName
        self.i = i
        if fName and (pData is None):
            self.ReadImage(fName)
        if bInit:
            self.pData = torch.FloatTensor(nLines,nCols)
        self.nvPad = 0
        self.nhPad = 0

    def clone(self):
        cl = CImage(self.nLines, self.nCols, self.pData.clone())
        return cl
    
    def IsPadded(self):
        return self.nvPad > 0 or self.nhPad > 0

    def ReadImage(self, fName):
        self.fName = fName
        print(f'Reading image {self.fName}...')
        bSrcIsFloat = self.fName.find('.float.') > 0
        if bSrcIsFloat:
            image = np.memmap(self.fName, dtype='float32', mode='r').__array__()
        else:
            image = np.memmap(self.fName, dtype='int16', mode='r').__array__()
        
        if len(image) != self.nLines * self.nCols:
            print('<ReadImage> size error', fName)
            print(f'<ReadImage> len(image) {len(image)} != {self.nLines} self.nLines * {self.nCols} self.nCols')
            sys.exit()
        
        image = torch.from_numpy(image.copy())
        image = image.view(self.nLines,self.nCols)
        if bSrcIsFloat:
            self.pData = image
        else:
            self.pData = image.float()
        
    def SetNameForDump(self):
        sName = self.fName
        if self.nCols == self.nLines:
            sName = SetValue(sName, '_matrix', self.nCols)
        else:
            sName = SetValue(sName, '_width', self.nCols)
            sName = SetValue(sName, '_height', self.nLines)
        sName = SetType(sName, '.float.rimg')
        self.fName = sName
    
    def SetNewNameForDump(self, sName):
        sName = SetValue(sName, '_width', self.nCols)
        sName = SetValue(sName, '_height', self.nLines)
        sName = SetType(sName, '.float.rimg')
        return sName
    
    def WriteToFile(self, fileName):
        npimage = self.pData.numpy()
        fullName = 'd:/Dump/' + fileName
        self.fName = fullName
        self.SetNameForDump()
        with open (fullName, 'wb') as file:
            file.write(npimage.tobytes())
        print('Image saved:', fullName)
    
    def WriteVariantToFile(self, pNewData, sfName):
        npimage = pNewData.numpy()
        if sfName[1] != ':':
            sfName = 'd:/Dump/' + sfName
        sfName = self.SetNewNameForDump(sfName)
        with open (sfName, 'wb') as file:
            file.write(npimage.tobytes())
        print('Image saved:', sfName)

    def WriteTensorDataToFile(self, fileName, pData, matrix):
        #print(f'<WriteTensorDataToFile> {fileName=}')
        npimage = pData.numpy()
        sName = SetValue(fileName, '_matrix', matrix)
        sName = SetType(sName, '.float.rimg')
        with open (sName, 'wb') as file:
            file.write(npimage.tobytes())
        print('Image saved:', sName)

    def WriteToFilePath(self, filePath = None, sDesc = None):
        if sDesc:
            print(f'<WriteToFilePath> sDesc: {sDesc}')
        else:
            print('<WriteToFilePath>')
        if not filePath is None:
             self.fName = filePath
        self.SetNameForDump()
        if sDesc:
            self.fName = AddDesc(self.fName, sDesc)
        npimage = self.pData.numpy()
        with open (self.fName, 'wb') as file:
            file.write(npimage.tobytes())
        print('Image saved:', self.fName)
        return self.fName

    def Show(self):
        plt.matshow(self.pData, cmap='gray')
        plt.show()
        
    def CreateCopy(self):
        myCopy = CImage(self.nLines, self.nCols, self.pData.clone())
        myCopy.fName = self.fName
        return myCopy
    
    def CreateUnPaddedCopy(self):
        if not self.IsPadded():
            return self.CreateCopy()
        
        nNewLines = self.nLines - 2 * self.nvPad
        nNewCols = self.nCols - 2 * self.nhPad
        newData = self.pData[self.nvPad:self.nLines-self.nvPad, self.nhPad:self.nCols-self.nhPad].clone()
        myCopy = CImage(nNewLines, nNewCols, newData)
        sImageName = self.fName
        sImageName = RemoveValue(sImageName, '_hPad')
        sImageName = RemoveValue(sImageName, '_vPad')
        sImageName = SetValue(sImageName, '_width', nNewCols)
        sImageName = SetValue(sImageName, '_height', nNewLines)
        myCopy.fName = sImageName
        return myCopy
    
    def Multiply(self, multiplier):
        self.pData = self.pData * multiplier
        
    def Subtract(self, other):
        self.pData = self.pData - other.pData
        
    def SetConstant(self, value):
        self.pData.fill_(value)
        
    def AddConstantVertically(self, iFromCol, value):
        for iLine in range(self.nLines):
            self.pData[iLine,iFromCol:].add_(value)
            
    def AddConstantToLine(self, iLine, value):
        self.pData[iLine,:].add_(value)
        
    def ZeroFirstLines(self, nLines):
        self.pData[0:nLines,:] = 0
            
    def ZeroLastLines(self, iFrom):
        self.pData[iFrom:,:] = 0
            
    def Print(self):
        print(f'<CImage:Print> name: {self.fName}, i: {self.i}')
        print(f'pData.size {self.pData.size()}')
        if self.nLines * self.nCols < 100:
            print('pData', self.pData)
        print('---')

def WriteImage(image, fPath):
    npimage = image.numpy()
    with open (fPath, 'wb') as file:
        file.write(npimage.tobytes())
    print('Image saved:', fPath)

def testReadAndWrite():
    origImages = ReadImages('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - NoANR/cf483db6_0_Recon0.raw')
    print('pimage.size()', origImages.size())
    nImages = len(origImages)
    origCentral = origImages[int(nImages/2),:,:] 
    print('nImages {}, image size {}'.format(nImages, origCentral.size()))
    WriteImage(origCentral, 'Try/central', 512, 512)

    plt.matshow(origCentral, cmap='gray')
    plt.show()  

    targetImages = ReadImages('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - ANR5/cf483db6_0_Recon0_ANR5.raw')
    targetCentral = targetImages[int(nImages/2),:,:] 

    plt.matshow(targetCentral, cmap='gray')
    plt.show()  

    diff = origCentral - targetCentral

    plt.matshow(diff, cmap='gray')
    plt.show()  
   
def TestCImage():
    image = CImage(512,512)
    image.Read1ImageOfVolume('F:/Data/Noise reduction DATA/SCANPLAN_7346_WideFOV_DFS/Offline_14_01_2021 - NoANR/cf483db6_0_Recon0.raw')
    image.WriteToFile('Try/centralFromClass')
    image.show()
    image2 = image.CreateCopy()
    image.Multiply(0.5)
    image.WriteToFile('Try/centralFromClass_halved')
    image.show()
    image2.WriteToFile('Try/centralFromClass_copy')
    image2.show()

def TestPatterns():
    image = CImage(4,10,bInit=True)
    image.SetConstant(10)
    print(image.__dict__)
    image.AddConstantVertically(5, 3)
    print(image.__dict__)
    
def TestStringValues():
    s = 'dd0c04c3_0_Recon0_Polar_width1152_height544_hPad16_vPad64_hsmooth51_SRC_im28.float.img'
    print ('string', s)
    print ('get value _im', GetValue(s,'_im'))
    print ('set value _im 3', SetValue(s,'_im', 3))
    print ('set value _im 456', SetValue(s,'_im', 456))
    print ('set value _width 22', SetValue(s,'_width', 22))
    print ('Remove value _width', RemoveValue(s,'_width'))
    print ('Remove value _im', RemoveValue(s,'_im'))
    

def main():
    print('Handle Images Main')
    #testReadAndWrite()
    #TestCImage()
    #TestPatterns()
    TestStringValues()

if __name__ == '__main__':
    main()

