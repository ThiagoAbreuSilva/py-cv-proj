#
# Autor: Thiago Abreu da Silva
# E-mail: thiago.abreu.84@gmail.com
# https://www.linkedin.com/in/thiago-abreu-da-silva/
#
# Deenvolvido em Python 3.5, Opencv 3.1.0-dev
#
#
#
#
#

# Circle Detection - Hough Cirlces
# Editado por TAS

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt
# %matplotlib inline

erosion_steps=False
intermediate_steps=True

def scale1channel(input):
    output = np.copy(input)
    minPixel = np.amin(output)
    output = output - minPixel*np.ones(input.shape[:2], dtype = "float")
    maxPixel = np.amax(output)
    output = (255/maxPixel)*(output)
    output = output.astype(np.uint8)
    return output

def scaleKernel1(input):
    output = np.copy(input)*100

    output = output.astype(np.int)

    return output

def scaleKernel2(input):
    output = np.copy(input)

    minPixel = np.amin(output)
    maxPixel = np.amax(output)

    print("minPixel = ", minPixel)
    print("maxPixel = ", maxPixel)

    absMax=1

    if(abs(maxPixel)>abs(minPixel)):
        absMax=abs(maxPixel)
    else:
        absMax=abs(minPixel)

    output = output*255/absMax

    output = output.astype(np.int)

    return output


def cropKernel(input_kernel, crop_size):

    # OBS: input_kernel sempre deve ter dimensao impar para largura e altura
    # OBS: crop_size sempre deve ter dimensao impar para largura e altura

    output_kernel=np.empty([crop_size, crop_size], input_kernel.dtype)


    xc = int(input_kernel.shape[0]/2)
    yc = int(input_kernel.shape[1]/2)

    limit = int(crop_size/2)

    output_kernel = input_kernel[xc - limit:xc+limit+1, yc-limit:yc+limit+1]

#     i, j= 0, 0
#     for line in output:
#         for element in line:
#             output[i,j]=round(element, n)

#     output=output*100
#     output = output.astype(np.int)

    return output_kernel


def applyLaplacianFilter(input):
    # Create our shapening kernel, we don't normalize since the
    # the values in the matrix sum to 1
    kernel_sharpening = np.array([[ 1, 1, 1],
                                  [ 1,-8, 1],
                                  [ 1, 1, 1]])

    # applying different kernels to the input image
    output = cv2.filter2D(input, -1, kernel_sharpening)
    return output

def getGaussianKernel2D(ksize, sigma):
    kx = cv2.getGaussianKernel(ksize, sigma)
    ky = cv2.getGaussianKernel(ksize, sigma)
    kernel_sharpening = kx*np.transpose(ky)
    return kernel_sharpening

def getDifferenceOfGaussiansKernel2D(ksize, sigma1, sigma2):
    # Create our shapening kernel, we don't normalize since the
    # the values in the matrix sum to 1

    # OBS: As 99.7% of the data are within 3 standard deviations of the mean.
    # OBS: Therefore the maximum accepted value for sigma is:
    # OBS: sigmaMAX = ksize/6

#     Mat kernelX = getGaussianKernel(kernelSize, sigmaX);
#     Mat kernelY = getGaussianKernel(kernelSize, sigmaY);
#     Mat kernelXY = kernelX * kernelY.t();

    kernel_sharpening1 = getGaussianKernel2D(ksize, sigma1)

#     kernel_sharpening1=scale1channel(kernel_sharpening1)
#     print("kernel1 = ", kernel_sharpening1)
#     cv2.waitKey(0)

    kernel_sharpening2 = getGaussianKernel2D(ksize, sigma2)

#     kernel_sharpening2=scale1channel(kernel_sharpening2)
#     print("kernel2 = ", kernel_sharpening2)
#     cv2.waitKey(0)

    kernel = kernel_sharpening1 - kernel_sharpening2

#     output = scaleKernel1(kernel)

#     print("kernel = ", kernel)
#     cv2.waitKey(0)

    return kernel

def applyKernelFilter(input, kernel):
    # applying different kernels to the input image
    output = cv2.filter2D(input, -1, kernel)
    return output




def applyDifferenceOfGaussians(input, ksize, sigma1, sigma2, crop_size=0):




    # The kernel below was calculated using difference of gaussians

#     kernel_sharpening = (1/100)*np.array([[ 0.00643822,  0.01569126,  0.01886257,  0.01569126,  0.00643822],
#                                           [ 0.01569126,  0.00679545, -0.03719233,  0.00679545,  0.01569126],
#                                           [ 0.01886257, -0.03719233, -0.1646693 , -0.03719233,  0.01886257],
#                                           [ 0.01569126,  0.00679545, -0.03719233,  0.00679545,  0.01569126],
#                                           [ 0.00643822,  0.01569126,  0.01886257,  0.01569126,  0.00643822]])


    kernel = getDifferenceOfGaussiansKernel2D(ksize, sigma1, sigma2)

    if(crop_size !=0):
        kernel = cropKernel(kernel, crop_size)

#     print(kernel)
#     cv2.waitKey(0)

    input_f = np.copy(input)
    input_f = input_f.astype(np.float)


#     kernel_sharpening = kernel_sharpening.astype(np.uint8)

    # applying different kernels to the input image
    output = cv2.filter2D(input_f, -1, kernel)

    output = output.astype(np.uint8)

    return output

def applyLaplacianFilter_GrayScale(input):

    output = applyLaplacianFilter(input)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = scale1channel(output)

    return output

def applyGaussianFilter(input, ksize):
    output = cv2.GaussianBlur(input, (ksize,ksize),0)
    output = scale1channel(output)
    return output

def invertImage(input):

    inputDict = getChannelDict1(input)

#     cv2.imshow('getChannelDict1[blue] antes', inputDict["blue"])
#     cv2.imshow('getChannelDict1[green] antes', inputDict["green"])
#     cv2.imshow('getChannelDict1[red] antes', inputDict["red"])

#     cv2.waitKey(0)

    inputDict["blue"] = 255 - inputDict["blue"]
    inputDict["green"] = 255 - inputDict["green"]
    inputDict["red"] = 255 - inputDict["red"]

#     cv2.imshow('getChannelDict1[blue] depois', inputDict["blue"])
#     cv2.imshow('getChannelDict1[green] depois', inputDict["green"])
#     cv2.imshow('getChannelDict1[red] depois', inputDict["red"])

#     cv2.waitKey(0)

    output = joinChannels2(inputDict["blue"], inputDict["green"], inputDict["red"])

    return output


def getChannelDict1(input):

    B = input[:,:,0]
    G = input[:,:,1]
    R = input[:,:,2]

#     B, G, R = cv2.split(buffer)
    # Let's create a matrix of zeros
    # with dimensions of the image h x w
#     zeros = np.zeros(input.shape[:2], dtype = "uint8")
    output = {}

    output["blue"] = B
    output["green"] = G
    output["red"] = R

    return output

def joinChannels1(B, G, R):

    height = B.shape[0]
    width = B.shape[1]

#     zeros = np.zeros((height, width, 3), dtype = "uint8")

    output = {}

    output = cv2.merge([B, G, R])

    return output

def joinChannels2(B, G, R, isImage=True):

    output = np.empty([B.shape[0], B.shape[1], 3])
#    output = np.zeros([B.shape[0], B.shape[1], 3])

    output[:,:,0] =  np.array(B[:,:])
    output[:,:,1] =  np.array(G[:,:])
    output[:,:,2] =  np.array(R[:,:])

    if(isImage):
        output = output.astype(np.uint8)

    return output

    output = {}

    output["blue"] = B
    output["green"] = G
    output["red"] = R

    return output

def plotImages(figIndex, suptitle, image1, image2):

    plt.figure(figIndex)
    plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.95,
                wspace=0.02, hspace=0)

    plt.subplot(121)
    # plt.axis("off")
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.imshow(image1)
    plt.subplot(122)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.imshow(image2)
    plt.suptitle(suptitle)
    plt.show()

    figIndex = figIndex + 1

    return figIndex

def getInitialImages(foldername):

    filenames = os.listdir(foldername)

    images = []

    for fname in filenames:
        filename = foldername + os.sep + fname
#         print(filename)
        images.append(cv2.imread(filename))

    return images

def applyChannelDict1PostOperations(B, G, R):

    B = B.astype(np.float16)
    G = G.astype(np.float16)
    R = R.astype(np.float16)

    B = (1/3)*B
    G = (1/3)*G
    R = (1/3)*R

    output = B + G + R

    if(True):
        output = scale1channel(output)
    else:
        output = output.astype(np.uint8)

    return output

def applyLaplacianFilter_GrayScaleEnhanced(input):
    channelImage = getChannelDict1(input)
    channelImage["blue"] = applyLaplacianFilter(channelImage["blue"])
    channelImage["green"] = applyLaplacianFilter(channelImage["green"])
    channelImage["red"] = applyLaplacianFilter(channelImage["red"])

    output = applyChannelDict1PostOperations(channelImage["blue"],
                                             channelImage["green"],
                                             channelImage["red"])
    return output

def applyErodeColoredImage(coloredImage, kernel, iterations ):

    channelImage = getChannelDict1(coloredImage)
    channelImage["blue"] = cv2.erode(channelImage["blue"], kernel, iterations)
    channelImage["green"] = cv2.erode(channelImage["green"], kernel, iterations)
    channelImage["red"] = cv2.erode(channelImage["red"], kernel, iterations)

    if(erosion_steps):
        cv2.imshow("channelImage[blue] antes", channelImage["blue"])
        cv2.imshow("channelImage[green] antes", channelImage["green"])
        cv2.imshow("channelImage[red] antes", channelImage["red"])

    channelImage["blue"] = scale1channel(channelImage["blue"])
    channelImage["green"] = scale1channel(channelImage["green"])
    channelImage["red"] = scale1channel(channelImage["red"])

    output = joinChannels2(channelImage["blue"], channelImage["green"], channelImage["red"])

    if(erosion_steps):
        cv2.imshow("channelImage[blue] depois", output[:,:,0])
        cv2.imshow("channelImage[green] depois", output[:,:,1])
        cv2.imshow("channelImage[red] depois", output[:,:,2])
        cv2.waitKey(0)

#         print("output[blue] = ", output[:,:,0])
#         print("output[green] = ", output[:,:,1])
#         print("output[red] = ", output[:,:,2])

    return output


def applyDilateColoredImage(coloredImage, kernel, iterations ):

    channelImage = getChannelDict1(coloredImage)
    channelImage["blue"] = cv2.dilate(channelImage["blue"], kernel, iterations)
    channelImage["green"] = cv2.dilate(channelImage["green"], kernel, iterations)
    channelImage["red"] = cv2.dilate(channelImage["red"], kernel, iterations)

    if(erosion_steps):
        cv2.imshow("channelImage[blue] antes", channelImage["blue"])
        cv2.imshow("channelImage[green] antes", channelImage["green"])
        cv2.imshow("channelImage[red] antes", channelImage["red"])

    channelImage["blue"] = scale1channel(channelImage["blue"])
    channelImage["green"] = scale1channel(channelImage["green"])
    channelImage["red"] = scale1channel(channelImage["red"])

    output = joinChannels2(channelImage["blue"], channelImage["green"], channelImage["red"])

    if(erosion_steps):
        cv2.imshow("channelImage[blue] depois", output[:,:,0])
        cv2.imshow("channelImage[green] depois", output[:,:,1])
        cv2.imshow("channelImage[red] depois", output[:,:,2])
        cv2.waitKey(0)

#         print("output[blue] = ", output[:,:,0])
#         print("output[green] = ", output[:,:,1])
#         print("output[red] = ", output[:,:,2])

    return output

def applyDifferenceOfGaussiansColoredImage(input, ksize, sigma1, sigma2, crop_size=0):
    channelImage = getChannelDict1(input)
    channelImage["blue"] = applyDifferenceOfGaussians(channelImage["blue"], ksize, sigma1, sigma2, crop_size=0)
    channelImage["green"] = applyDifferenceOfGaussians(channelImage["green"], ksize, sigma1, sigma2, crop_size=0)
    channelImage["red"] = applyDifferenceOfGaussians(channelImage["red"], ksize, sigma1, sigma2, crop_size=0)

    channelImage["blue"] = scale1channel(channelImage["blue"])
    channelImage["green"] = scale1channel(channelImage["green"])
    channelImage["red"] = scale1channel(channelImage["red"])

    output = joinChannels2(channelImage["blue"], channelImage["green"], channelImage["red"])

    return output

def applyGaussianFilterColoredImage(input, ksize):
    channelImage = getChannelDict1(input)
    channelImage["blue"] = applyGaussianFilter(channelImage["blue"], ksize)
    channelImage["green"] = applyGaussianFilter(channelImage["green"], ksize)
    channelImage["red"] = applyGaussianFilter(channelImage["red"], ksize)

    channelImage["blue"] = scale1channel(channelImage["blue"])
    channelImage["green"] = scale1channel(channelImage["green"])
    channelImage["red"] = scale1channel(channelImage["red"])

    output = joinChannels2(channelImage["blue"], channelImage["green"], channelImage["red"])

    return output


def applyDifferenceOfGaussians_GrayScaleEnhanced(input, ksize, sigma1, sigma2, crop_size=0):
    channelImage = getChannelDict1(input)
    channelImage["blue"] = applyDifferenceOfGaussians(channelImage["blue"], ksize, sigma1, sigma2, crop_size=0)
    channelImage["green"] = applyDifferenceOfGaussians(channelImage["green"], ksize, sigma1, sigma2, crop_size=0)
    channelImage["red"] = applyDifferenceOfGaussians(channelImage["red"], ksize, sigma1, sigma2, crop_size=0)

    output = applyChannelDict1PostOperations(channelImage["blue"],
                                             channelImage["green"],
                                             channelImage["red"])
    return output


def getMagnitudeImage(input):

    output=np.empty([input.shape[0], input.shape[1]], np.float)

    channelImage = getChannelDict1(input)

    channelImage["blue"] = channelImage["blue"].astype(np.float)
    channelImage["green"] = channelImage["green"].astype(np.float)
    channelImage["red"] = channelImage["red"].astype(np.float)

#     print("channelImage[blue].dtype", channelImage["blue"].dtype)
#     print("channelImage[green].dtype", channelImage["green"].dtype)
#     print("channelImage[red].dtype", channelImage["red"].dtype)

#     channelImage[0] = np.float(channelImage[0])
#     channelImage[1] = np.float(channelImage[1])
#     channelImage[2] = np.float(channelImage[2])

    channelImageSquared = np.empty([input.shape[0], input.shape[1], 3], np.float)

    channelImageSquared[:,:,0] = np.square(channelImage["blue"])

    channelImageSquared[:,:,1] = np.square(channelImage["green"])

    channelImageSquared[:,:,2] = np.square(channelImage["red"])

    output = channelImageSquared[:,:,0] + channelImageSquared[:,:,1] + channelImageSquared[:,:,2]

#     print("outputSquared.dtype = ",output.dtype)
#     print("np.amin(output) = ", np.amin(output))
#     print("np.amax(output) = ", np.amax(output))

    output = np.sqrt(output)

#     print("outputSqrt.dtype = ",output.dtype)
#     print("np.amin(output) = ", np.amin(output))
#     print("np.amax(output) = ", np.amax(output))

    output = scale1channel(output)

    output = np.uint8(np.around(output))

#     output = output.astype(np.uint8)

    return output






# iterations=1
# k_side=11
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
# # Now we erode
# erosion = cv2.erode(thresh, kernel, iterations)




foldername = "imagens"

initialImages = getInitialImages(foldername)

cv2.waitKey(0)

# image1 = initialImages[0]
# image2 = initialImages[1]

# print("Media Imagem 1 = ", np.average(image1))
# print("Media Imagem 2 = ", np.average(image2))

# f, axarr = plt.subplots(1, 2)
# axarr[0, 0].imshow(image1)
# axarr[0, 1].imshow(image2)
# f.suptitle('Initial Images', fontsize='large')

figIndex=1
figIndex = plotImages(figIndex,"Initial Images", initialImages[0], initialImages[1])

theshInv = 160
imagesInv = []
for img in initialImages:
    if(np.average(img)>theshInv):
        img = invertImage(img)
        imagesInv.append(img)
    else:
        imagesInv.append(img)

# print(imagesInv)

# image1Inv = invertImage(image1)
# image2Inv = invertImage(image2)

figIndex = plotImages(figIndex,"Inverted Images", imagesInv[0], imagesInv[1])

# increment = 10
# image1InvRaised = cv2.add(image1Inv, increment)
# image2InvRaised = cv2.add(image2Inv, increment)

# figIndex = plotImages(figIndex,"Raised Images", image1InvRaised, image2InvRaised)

iterations=1
ksize=11
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))

imagesEroded = []
for img in imagesInv:
    img = applyErodeColoredImage(img, kernel, iterations )
    imagesEroded.append(img)

figIndex = plotImages(figIndex,"Eroded Images", imagesEroded[0], imagesEroded[1])


imagesDilated = []
for img in imagesEroded:
    img = applyDilateColoredImage(img, kernel, iterations )
    imagesDilated.append(img)

figIndex = plotImages(figIndex,"Dilated Images", imagesDilated[0], imagesDilated[1])


# # edged=applyDifferenceOfGaussians(gray_inversed, ksize, 0.55*sigmaMax, 0.35*sigmaMax, 9 )
# edged=applyDifferenceOfGaussians(gray_inversed, ksize, 0.59*sigmaMax, 0.34*sigmaMax )

ksize=15
sigmaMax=ksize/6
imagesEdged = []
for img in imagesDilated:
#     img = applyDifferenceOfGaussians_GrayScaleEnhanced(img, ksize, 0.55*sigmaMax, 0.35*sigmaMax, crop_size=9)
    img = applyDifferenceOfGaussiansColoredImage(img, ksize, 0.57*sigmaMax, 0.34*sigmaMax)
    imagesEdged.append(img)

figIndex = plotImages(figIndex,"Edged Images", imagesEdged[0], imagesEdged[1])


ksize=9
imagesBlurred = []
for img in imagesEdged:
    img = applyGaussianFilterColoredImage(img, ksize)
#     img = img.astype(np.uint8)
    imagesBlurred.append(img)

figIndex = plotImages(figIndex,"Blurred Images", imagesBlurred[0], imagesBlurred[1])


imagesMagnitude = []
for img in imagesBlurred:
    img = getMagnitudeImage(img)
#     img = img.astype(np.uint8)
    imagesMagnitude.append(img)

figIndex = plotImages(figIndex,"Color Magnitude of Images", imagesMagnitude[0], imagesMagnitude[1])


# print("imagesMagnitude[0].dtype = ",imagesMagnitude[0].dtype)
# print("np.amin(imagesMagnitude[0]) = ", np.amin(imagesMagnitude[0]))
# print("np.amax(imagesMagnitude[0]) = ", np.amax(imagesMagnitude[0]))
# print("imagesMagnitude[1].dtype = ",imagesMagnitude[1].dtype)
# print("np.amin(imagesMagnitude[1]) = ", np.amin(imagesMagnitude[1]))
# print("np.amax(imagesMagnitude[1]) = ", np.amax(imagesMagnitude[1]))

########################################################################

minR = 6
centersMinDist = 2*minR

circlesOfImages = []
for img in imagesMagnitude:

    circles = cv2.HoughCircles(img,
                           cv2.HOUGH_GRADIENT,
                           1, #1.53,
                           centersMinDist,
                           param1 = 150 ,
                           param2 = 70, #param2 = 150 ,
                           minRadius=minR)

    circles = np.uint16(np.around(circles))

    circlesOfImages.append(circles)

finalImages = []
for circles, image  in zip(circlesOfImages, initialImages):
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 5)
        finalImages.append(image)

figIndex = plotImages(figIndex,"Detected Circles", finalImages[0], finalImages[1])

########################################################################

cv2.waitKey(0)
cv2.destroyAllWindows()
