# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:15:27 2019

@author: Haris
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
#np.set_printoptions(threshold=np.inf)
                                
while(True):
        n=int(input("Enter 1 for gray, Enter 2 for RGB, or Enter 0 for the exit:\n"))
        
        if(n==0):
                
              print("Exiting program...")
              break
      
        elif(n==1): 
                
                print("Working on it...")
                
                imageFace1=cv2.imread('6_3_s.bmp')
                grayImageFace1 = cv2.cvtColor(imageFace1, cv2.COLOR_BGR2GRAY)
                b,g,r=imageFace1[:,:,0],imageFace1[:,:,1],imageFace1[:,:,2]
                grayImageFace0 = 0.114* b + 0.587* g + 0.299* r
                grayImageFace1=np.rint(grayImageFace0)
                grayImageFace2=np.int_(grayImageFace1)
                #grayImageFace = cv2.cvtColor(imageFace1, cv2.COLOR_BGR2GRAY)
                imageFace1=grayImageFace2
                #imageFace2= cv2.GaussianBlur(grayImageFace,(3,3),5)
               
                imageTree1=cv2.imread('2_13_s.bmp')
                b,g,r=imageTree1[:,:,0],imageTree1[:,:,1],imageTree1[:,:,2]
                grayImageTree0 = 0.114* b + 0.587* g + 0.299* r
                grayImageTree1=np.rint(grayImageTree0)
                grayImageTree2=np.int_(grayImageTree1)
                #grayImageTree = cv2.cvtColor(imageTree1, cv2.COLOR_BGR2GRAY)
                imageTree1=grayImageTree2
                #imageTree1=cv2.GaussianBlur(grayImageTree,(3,3),5)
                
                
                #This is actually our Kernel(Filter)
                Gx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
                Gy=np.transpose(Gx)
                
                #Now, we are working with NxM dimension images, not with NxMx3 dimension images, because our images are in grayscale mode
                NImageFace=imageFace1.shape[0]
                MImageFace=imageFace1.shape[1]
                
                NImageTree=imageTree1.shape[0]
                MImageTree=imageTree1.shape[1]
                
                
                rowsImageFace=NImageFace-2
                columnsImageFace=MImageFace-2
                
                rowsImageTree=NImageTree-2
                columnsImageTree=MImageTree-2
                
                
                resultXImageFace=np.zeros((rowsImageFace,columnsImageFace))
                resultYImageFace=np.zeros((rowsImageFace,columnsImageFace))
                finalResultImageFace=np.zeros((rowsImageFace,columnsImageFace))
                
                resultXImageTree=np.zeros((rowsImageTree,columnsImageTree))
                resultYImageTree=np.zeros((rowsImageTree,columnsImageTree))
                finalResultImageTree=np.zeros((rowsImageTree,columnsImageTree))
                
                
                #First check if these two images have the same shape
                #Formats can be: (320,213) or (213,320)             
                if(imageFace1.shape==imageTree1.shape):
                        
                        for i in range(rowsImageFace):
                            for j in range(columnsImageFace):
                                for l in range(len(Gx)):
                                    for k in range(len(Gy)):
                                        resultXImageFace[i][j]+=Gx[l][k]*imageFace1[i+l][j+k]
                                        resultYImageFace[i][j]+=Gy[l][k]*imageFace1[i+l][j+k]
                                        
                                        resultXImageTree[i][j]+=Gx[l][k]*imageTree1[i+l][j+k]
                                        resultYImageTree[i][j]+=Gy[l][k]*imageTree1[i+l][j+k]
                                        
                                        
                                finalResultImageFace[i][j] = np.sqrt(resultXImageFace[i][j] **2 + resultYImageFace[i][j] **2)
                                finalResultImageTree[i][j] = np.sqrt(resultXImageTree[i][j] **2 + resultYImageTree[i][j] **2)
                                
                        cv2.imwrite('grayImageFaceX_0.jpg',resultXImageFace)
                        cv2.imwrite('grayImageFaceY_0.jpg',resultYImageFace)
                        cv2.imwrite('grayImageFaceFinal_0.jpg',finalResultImageFace)
                        cv2.imwrite('grayImageTreeX_0.jpg',resultXImageTree)
                        cv2.imwrite('grayImageTreeY_0.jpg',resultYImageTree)
                        cv2.imwrite('grayImageTreeFinal_0.jpg',finalResultImageTree)
                        print("Process is done! Please check your folder in order to see the images")
                        break
                        
                else:
                        
                        for i in range(rowsImageFace):
                            for j in range(columnsImageFace):
                                for l in range(len(Gx)):
                                    for k in range(len(Gy)):
                                        resultXImageFace[i][j]+=Gx[l][k]*imageFace1[i+l][j+k]
                                        resultYImageFace[i][j]+=Gy[l][k]*imageFace1[i+l][j+k]
                                        
                                        resultXImageTree[j][i]+=Gx[l][k]*imageTree1[j+l][i+k]
                                        resultYImageTree[j][i]+=Gy[l][k]*imageTree1[j+l][i+k]
                                        
                                        
                                finalResultImageFace[i][j] = np.sqrt(resultXImageFace[i][j] **2 + resultYImageFace[i][j] **2)
                                finalResultImageTree[j][i] = np.sqrt(resultXImageTree[j][i] **2 + resultYImageTree[j][i] **2)
                        
                        cv2.imwrite('grayImageFaceX_0.jpg',resultXImageFace)
                        cv2.imwrite('grayImageFaceY_0.jpg',resultYImageFace)
                        cv2.imwrite('grayImageFaceFinal_0.jpg',finalResultImageFace)
                        cv2.imwrite('grayImageTreeX_0.jpg',resultXImageTree)
                        cv2.imwrite('grayImageTreeY_0.jpg',resultYImageTree)
                        cv2.imwrite('grayImageTreeFinal_0.jpg',finalResultImageTree)
                        print("Process is done! Please check your folder in order to see the images")
                        break
                
                                              
                        #Part for checking if the matrices are the same in the second case when the shapes of the images are not same
#                        resultXImageTree1=np.zeros((rowsImageTree,columnsImageTree))
#                        resultYImageTree1=np.zeros((rowsImageTree,columnsImageTree))
#                        finalResultImageTree1=np.zeros((rowsImageTree,columnsImageTree))
#                        
#                        for i in range(0,rowsImageTree):
#                                for j in range(0, columnsImageTree):
#                                        for l in range(0,len(Gx)):
#                                                for k in range(0,len(Gy)):
#                                                        resultXImageTree1[i][j]+=Gx[l][k]*imageTree1[i+l][j+k]
#                                                        resultYImageTree1[i][j]+=Gy[l][k]*imageTree1[i+l][j+k]
#                        
#                        
#                                        finalResultImageTree1[i][j] = np.sqrt(resultXImageTree1[i][j] **2 + resultYImageTree1[i][j] **2)
#                        
#                        
#                        
#                        def same(A,B):
#                                
#                                rowsOfA=A.shape[0]
#                                columnsOfA=A.shape[1]
#                        
#                                rowsOfB=B.shape[0]
#                                columnsOfB=B.shape[1]
#                                     
#                                if(rowsOfA==rowsOfB and columnsOfA==columnsOfB):    
#                                        for i in range(rowsOfA):
#                                                for j in range(columnsOfA):
#                                                        if(A[i][j]!=B[i][j]):
#                                                                print("Matrices are not same!")
#                                                                return 0
#                                                        
#                                        print("Matrices are the same!")
#                                        return 1
#                        
#                        
#                        same(finalResultImageTree,finalResultImageTree1)    
                        

        elif(n==2):
                
                print("Working on it...")
              
                imageFace1=cv2.imread('6_3_s.bmp')
                imageTree1=cv2.imread('2_13_s.bmp')
                
                Gx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
                Gy=np.transpose(Gx)
                
                
                NImageFace=imageFace1.shape[0]
                MImageFace=imageFace1.shape[1]
                KImageFace=imageFace1.shape[2]
                
                NImageTree=imageTree1.shape[0]
                MImageTree=imageTree1.shape[1]
                KImageTree=imageTree1.shape[2]
                
             
                rowsImageFace=NImageFace-2
                columnsImageFace=MImageFace-2
                channelImageFace=KImageFace
                
                rowsImageTree=NImageTree-2
                columnsImageTree=MImageTree-2
                channelImageTree=KImageTree
                
              
                resultXImageFace=np.zeros((rowsImageFace,columnsImageFace))
                resultYImageFace=np.zeros((rowsImageFace,columnsImageFace))
                finalResultImageFace=np.zeros((rowsImageFace,columnsImageFace))
                
                resultXImageTree=np.zeros((rowsImageTree,columnsImageTree))
                resultYImageTree=np.zeros((rowsImageTree,columnsImageTree))
                finalResultImageTree=np.zeros((rowsImageTree,columnsImageTree))
        
                if(imageFace1.shape==imageTree1.shape):
                        
                      for i in range(rowsImageFace):
                                for j in range(columnsImageFace):
                                        for z in range(channelImageFace):
                                                for l in range(channelImageFace):
                                                        for k in range(len(Gx)):
                                                                resultXImageFace[i][j]+=Gx[l][k]*imageFace1[i+l][j+k][z]
                                                                resultYImageFace[i][j]+=Gy[l][k]*imageFace1[i+l][j+k][z]
                                                                
                                                                resultXImageTree[j][i]+=Gx[l][k]*imageTree1[j+l][i+k][z]
                                                                resultYImageTree[j][i]+=Gy[l][k]*imageTree1[j+l][i+k][z]
                                                
                                                
                                        finalResultImageFace[i][j] = np.sqrt(resultXImageFace[i][j] **2 + resultYImageFace[i][j] **2)
                                        finalResultImageTree[j][i] = np.sqrt(resultXImageTree[j][i] **2 + resultYImageTree[j][i] **2)
                                        
                      cv2.imwrite('BGRImageFaceX_0.jpg',resultXImageFace)
                      cv2.imwrite('BGRImageFaceY_0.jpg',resultYImageFace)
                      cv2.imwrite('BGRImageFaceFinal_0.jpg',finalResultImageFace)
                      cv2.imwrite('BGRImageTreeX_0.jpg',resultXImageTree)
                      cv2.imwrite('BGRImageTreeY_0.jpg',resultYImageTree)
                      cv2.imwrite('BGRImageTreeFinal_0.jpg',finalResultImageTree)
                      print("Process is done! Please check your folder in order to see the images")
                      break
                                          
                else:
                
                        for i in range(rowsImageFace):
                                for j in range(columnsImageFace):
                                        for z in range(channelImageFace):
                                                for l in range(channelImageFace):
                                                        for k in range(len(Gx)):
                                                                resultXImageFace[i][j]+=Gx[l][k]*imageFace1[i+l][j+k][z]
                                                                resultYImageFace[i][j]+=Gy[l][k]*imageFace1[i+l][j+k][z]
                                                                
                                                                resultXImageTree[j][i]+=Gx[l][k]*imageTree1[j+l][i+k][z]
                                                                resultYImageTree[j][i]+=Gy[l][k]*imageTree1[j+l][i+k][z]
                                                
                                                
                                        finalResultImageFace[i][j] = np.sqrt(resultXImageFace[i][j] **2 + resultYImageFace[i][j] **2)
                                        finalResultImageTree[j][i] = np.sqrt(resultXImageTree[j][i] **2 + resultYImageTree[j][i] **2)
                                        
                        cv2.imwrite('BGRImageFaceX_0.jpg',resultXImageFace)
                        cv2.imwrite('BGRImageFaceY_0.jpg',resultYImageFace)
                        cv2.imwrite('BGRImageFaceFinal_0.jpg',finalResultImageFace)
                        cv2.imwrite('BGRImageTreeX_0.jpg',resultXImageTree)
                        cv2.imwrite('BGRImageTreeY_0.jpg',resultYImageTree)
                        cv2.imwrite('BGRImageTreeFinal_0.jpg',finalResultImageTree)
                        print("Process is done! Please check your folder in order to see the images")
                        break
                
               
               
        else:
                print(n=int(input("\nPlease Enter 1 for gray, Enter 2 for RGB or Enter 0 for exit\n")))
        