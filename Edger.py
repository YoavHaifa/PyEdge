# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:45:47 2021
Use Sobel for Edge Detection to use in input and/or loss function

@author: yoav.bar
"""

import torch
import torch.nn.functional as F
#import sys

from HandleImages import CImage

debug = 0

sfImageName1 = './Cardiac_example_matrix512.float.TImage'
sfImageName2 = './Phantom_example_matrix512.short.TImage'

def MatNorm(mat):
    absMat = torch.abs(mat)
    sumAbs = torch.sum(absMat)
    mat = mat * 10 / sumAbs
    return mat

class CEdger():
    
    def __init__(self, bTest=False, debug=0):

        #if args.device:
        #    self.device = torch.device(args.device)
        #else:
        
        if bTest:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if debug:
            print('++++++')
            print(f'<CEdger::__init__> device {self.device}')
            print('++++++')
            
        kernel_size = 3
        sobel_mat = torch.zeros(size=(4, 1, kernel_size, kernel_size))

        kernel_mid = kernel_size // 2
        for idx in range(4):
            if idx == 0:
                sobel_mat[idx, :, 0, :] = -1
                sobel_mat[idx, :, 0, kernel_mid] = -2
                sobel_mat[idx, :, -1, :] = 1
                sobel_mat[idx, :, -1, kernel_mid] = 2
            elif idx == 1:
                sobel_mat[idx, :, :, 0] = -1
                sobel_mat[idx, :, kernel_mid, 0] = -2
                sobel_mat[idx, :, :, -1] = 1
                sobel_mat[idx, :, kernel_mid, -1] = 2
            elif idx == 2:
                sobel_mat[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    sobel_mat[idx, :, kernel_mid - i, i] = -1
                    sobel_mat[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                sobel_mat[idx, :, -1, -1] = 2
            else:
                sobel_mat[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    sobel_mat[idx, :, kernel_mid + i, i] = -1
                    sobel_mat[idx, :, i, kernel_mid + i] = 1
                sobel_mat[idx, :, 0, -1] = 2
        
        
        for i in range(4):
            sobel_mat[i,0] = MatNorm(sobel_mat[i,0])
            
        self.sobelMat = sobel_mat
        #print('self.sobelMat.type()', self.sobelMat.type())
        self.sobelMat = self.sobelMat.to(self.device)
        if debug:
            print('self.sobelMat.type()', self.sobelMat.type())

        # Prepare smooth filter
        filt = torch.ones(1, 1, 3, 3) / 9
        if debug:
            print('smoothFilt 3 * 3', filt)
        self.smoothFilt = filt.to(self.device)

    def Print(self):
        print(f'<Edger::Print> self.sobelMat shape {self.sobelMat.shape}')
        print(f'sobel_mat {self.sobelMat}')
        
    def Compute(self, im):
        im1 = torch.unsqueeze(im, 0)
        im1 = torch.unsqueeze(im1, 0)
        out = F.conv2d(im1, self.sobelMat)
        return out
        
    def BComputeMax(self, bim):
        #print(f'<BComputeMax> bim {bim}')
        #print(f'<BComputeMax> self.sobelMat {self.sobelMat}')
        bout = F.conv2d(bim, self.sobelMat)
        bout = torch.abs(bout)
        boutMax, _ = torch.max(bout, 1)
        """
        if debug:
            print('<BComputeMax> bim shape', bim.shape)
            print('bout shape', bout.shape)
            print('bout range', torch.min(bout), torch.max(bout))
            print('boutMax shape', boutMax.shape)
            print('')
            """
        return boutMax
        
    def BComputeMax1(self, bim):
        bimShape = bim.shape
        if debug:
            print(f'<BComputeMax1> bim shape {bimShape}')
        nCols = bimShape[-1]
        nLines = bimShape[-2]
        bout = F.conv2d(bim, self.sobelMat)
        bout = torch.abs(bout)
        boutMax, _ = torch.max(bout, 1)
        if debug:
            print(f'<BComputeMax1> bout shape {bout.shape}')

        edgeMax1 = torch.zeros_like(bim)
        edgeMax1[:,:,1:nLines-1,1:nCols-1] = boutMax.unsqueeze(1)

        if debug:
            print('<BComputeMax> bim shape', bim.shape)
            print('bout shape', bout.shape)
            print('bout range', torch.min(bout), torch.max(bout))
            print('boutMax shape', boutMax.shape)
            print('')

        return edgeMax1
        
    def BComputeMaxSmooth(self, bim, nSmooth=1):
        if debug:
            print(f'<BComputeMaxSmooth> bim shape {bim.shape}, nSmooth {nSmooth}')
        edgeSmoothed = self.BComputeMax1(bim)
        for i in range(nSmooth):
            edgeSmoothed = torch.nn.functional.conv2d(edgeSmoothed, self.smoothFilt, padding='same')
        return edgeSmoothed
        
    def BComputeMax2(self, bim):
        bout = F.conv2d(bim, self.sobelMat, padding=(1,1), stride=(1,1), padding_mode='reflect', bias=False)
        #bout = F.conv2d(bim, self.sobelMat, padding=1)
        #bout = F.conv2d(bim, self.sobelMat, padding=1, padding_mode='replicate')
        bout = torch.abs(bout)
        boutMax, _ = torch.max(bout, 1)
        """
        if debug:
            print('<BComputeMax> bim shape', bim.shape)
            print('bout shape', bout.shape)
            print('bout range', torch.min(bout), torch.max(bout))
            print('boutMax shape', boutMax.shape)
            print('')
            """
        return boutMax
    
    def Smooth(self, images):
        return torch.nn.functional.conv2d(images, self.smoothFilt, padding='same')


def TestEdger(edger, sName, bSmooth=True):
    print(f'<TestEdger> {sName=}, {bSmooth=}')
    bMargins = True
    #bSmooth = True
    nSmooth = 3
    image1 = CImage(fName=sfImageName1)
    #print('image1', image1)
    #print('image dict', image1.__dict__)
    
    image2 = CImage(fName=sfImageName2)
    
    bim = torch.zeros([2,1,512,512])
    bim[0,0] = image1.pData;
    bim[1,0] = image2.pData;
    
    print('bim.shape', bim.shape)
    if bMargins:
        if bSmooth:
            edgeMax1 = edger.BComputeMaxSmooth(bim, nSmooth=nSmooth)
        else:
            edgeMax1 = edger.BComputeMax1(bim)
    else:
        edgeMax1 = edger.BComputeMax2(bim).unsqueeze(1)
        
    print('edgeMax.shape', edgeMax1.shape)
    
    matrix = len(edgeMax1[0,0])
    if bSmooth:
        image1.WriteTensorDataToFile(f'Dump/Cardiac_Smooth{nSmooth}_edgeMax1.rimg', edgeMax1[0], matrix)
        image2.WriteTensorDataToFile(f'Dump/Phantom_Smooth{nSmooth}_edgeMax1.rimg', edgeMax1[1], matrix)
    else:
        image1.WriteTensorDataToFile('Dump/Cardiac_edgeMax1.rimg', edgeMax1[0], matrix)
        image2.WriteTensorDataToFile('Dump/Phantom_edgeMax1.rimg', edgeMax1[1], matrix)


def main():
    edger = CEdger(bTest=True)
    TestEdger(edger, 'Sobel3_max', bSmooth=False)
    TestEdger(edger, 'Sobel3_max', bSmooth=True)
    #edger5 = CSobel5(bTest=True)
    #TestEdger(edger5, 'Sobel5')


if __name__ == '__main__':
    main()
    