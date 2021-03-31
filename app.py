import tkinter as tk
import tkinter as ttk
from tkinter import *
from tkinter import filedialog, Text
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import cv2
import scipy
from skimage.filters import threshold_otsu
from skimage import color
from skimage import io
import numpy as np
from tkinter.font import Font
import csv
from skimage.filters import threshold_local
from skimage.filters import try_all_threshold
from skimage.filters import sobel
from skimage.filters import gaussian
from skimage import exposure
from skimage import morphology
import os
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.restoration import inpaint


image1=''
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cwd = os.getcwd()
        print(cwd)
        os.chdir(dir_path)
        self.bold_font = Font(family="Helvetica", size=14, weight="bold")
        self.title("Python Image Processing")
        self.minsize(1350, 700)
        self.maxsize(1350, 700)
        self.labelFrame = ttk.Label(self, text = "Open File:", font=('Impact', -20), bg='#000', fg='#fff')
        self.labelFrame.grid(column = 0, row = 2, padx = 20, pady = 20)
        self.configure(background='#263D42')
        self.labelFrame.configure(background='#263D42')
        self.button()
        self.showoriginal()
        self.showgray()
        self.showotsu()
        self.redhist()
        self.whitehist()
        self.local1()
        self.otherthresh()
        self.edgedetection()
        self.gaussianblur()
        self.histequ()
        self.adaptequ()
        self.erosionshow()
        self.dilatedshow()
        self.logoremove()
        self.makenoise()
        self.tv()
        self.bilateral()
        self.slic()
        self.contour()


    def contour(self):
            self.contour= ttk.Button(self, text = "Contour(show)",command = self.showcontour)
            self.contour.configure(background='#e28743')
            self.contour.grid(column = 3, row = 3)

    def slic(self):
            self.slic= ttk.Button(self, text = "SLIC(segmentation)",command = self.slicsegmentation)
            self.slic.configure(background='#e28743')
            self.slic.grid(column = 2, row = 8)

    def bilateral(self):
            self.bilateral= ttk.Button(self, text = "Bilateral(Denoise)",command = self.bilateralnoise)
            self.bilateral.configure(background='#e28743')
            self.bilateral.grid(column = 2, row = 7)

    def tv(self):
            self.tv= ttk.Button(self, text = "Total Variation(Denoise)",command = self.tvnoise)
            self.tv.configure(background='#e28743')
            self.tv.grid(column = 2, row = 6)

    def makenoise(self):
            self.makenoise= ttk.Button(self, text = "Noisy(Random noise)",command = self.noisy)
            self.makenoise.configure(background='#e28743')
            self.makenoise.grid(column = 2, row = 5)

    def logoremove(self):
            self.logoremove= ttk.Button(self, text = "restoration(Logo Remove, depend on mask)",command = self.logo)
            self.logoremove.configure(background='#e28743')
            self.logoremove.grid(column = 2, row = 4)

    def dilatedshow(self):
            self.dilatedshow= ttk.Button(self, text = "morphology(Dilation)",command = self.dilation)
            self.dilatedshow.configure(background='#e28743')
            self.dilatedshow.grid(column = 2, row = 3)

    def erosionshow(self):
            self.erosionshow= ttk.Button(self, text = "morphology(Erosion)",command = self.erosion)
            self.erosionshow.configure(background='#e28743')
            self.erosionshow.grid(column = 1, row = 8)

    def adaptequ(self):
            self.adaptequ= ttk.Button(self, text = "Adaptive Equalization",command = self.adaptequal)
            self.adaptequ.configure(background='#e28743')
            self.adaptequ.grid(column = 1, row = 7)

    def histequ(self):
            self.histequ= ttk.Button(self, text = "Histogram Equalization",command = self.histequal)
            self.histequ.configure(background='#e28743')
            self.histequ.grid(column = 1, row = 6)

    def gaussianblur(self):
            self.gaussianblur= ttk.Button(self, text = "Gaussian Blur",command = self.gauss)
            self.gaussianblur.configure(background='#e28743')
            self.gaussianblur.grid(column = 1, row = 5)

    def button(self):
            self.button = ttk.Button(self, text = "Browse File",font=('Impact', -10),command = self.fileDialog)
            self.button.grid(column=1, row = 2)


    def showotsu(self):
            self.showotsu= ttk.Button(self, text = "Global thresholding",command = self.otsu)
            self.showotsu.configure(background='#e28743')
            self.showotsu.grid(column = 0, row = 5)

    def showgray(self):
            self.showgray = ttk.Button(self, text = "Grayscale",command = self.gray)
            self.showgray.configure(background='#e28743')
            self.showgray.grid(column = 0, row = 4)

    def showoriginal(self):
            self.showoriginal = ttk.Button(self, text = "Original",command = self.original)
            self.showoriginal.configure(background='#e28743')
            self.showoriginal.grid(column= 0, row = 3)

    def redhist(self):
            self.redhist= ttk.Button(self, text = "Histogram of RGB",command = self.red)
            self.redhist.configure(background='#e28743')
            self.redhist.grid(column = 0, row = 6)

    def whitehist(self):
            self.whitehist= ttk.Button(self, text = "Histogram of B&W",command = self.white)
            self.whitehist.configure(background='#e28743')
            self.whitehist.grid(column = 0, row = 7)

    def local1(self):
            self.local1= ttk.Button(self, text = "Local thresholding",command = self.localt)
            self.local1.configure(background='#e28743')
            self.local1.grid(column = 0, row = 8)

    def otherthresh(self):
            self.otherthresh= ttk.Button(self, text = "Other thresholdings",command = self.others)
            self.otherthresh.configure(background='#e28743')
            self.otherthresh.grid(column = 1, row = 3)

    def edgedetection(self):
            self.edgedetection= ttk.Button(self, text = "Edge Detection",command = self.edge)
            self.edgedetection.configure(background='#e28743')
            self.edgedetection.grid(column = 1, row = 4)



    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        #self.label = ttk.Label(self.labelFrame, text = "")

        #self.label.configure(text = self.filename)
        img = Image.open(self.filename)
        file="imageprocessing.csv"
        with open(file, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([self.filename])
        img2 = io.imread(self.filename, plugin='matplotlib')


    def otsu(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)
        image = color.rgb2gray(image)
        image=image*100
        print(image)
        thresh = threshold_otsu(image)
        binary_global=image
        binary_global=np.where(binary_global>thresh,255,0)
        show_image(binary_global, 'Global thresholding(Otsu)')

    def original(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = cv2.imread(image1)
        image = mpimg.imread(image1)
        plt.imshow(image)
        plt.show()


    def gray(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)#
        image = color.rgb2gray(image)
        show_image(image, 'Grayscale image')

    def red(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)
        plt.figure(figsize=(20,8),edgecolor='blue')
        plt.subplot(1, 3, 1)
        red_channel = image[:, :, 0]
        blue_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        # Plot the red histogram with bins in a range of 256
        plt.hist(red_channel.ravel(), bins=256)
        plt.title('Red Histogram')
        plt.xlabel('Labels', fontsize=10)
        plt.ylabel('Number of Pixels', fontsize=10)
        plt.subplot(1, 3, 2)
        plt.hist(green_channel.ravel(), bins=256)
        plt.title('Green Histogram')
        plt.subplot(1, 3, 3)
        plt.hist(blue_channel.ravel(), bins=256)
        plt.title('Blue Histogram')

        # Set title and show
        plt.show()

    def white(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)
        image = color.rgb2gray(image)
        plt.figure(figsize=(20,8),edgecolor='blue')
        plt.subplot(1, 1, 1)
        red_channel = image[:, :]
        # Plot the red histogram with bins in a range of 256
        plt.hist(red_channel.ravel(), bins=256)
        plt.title('Gray Histogram')
        plt.xlabel('Labels', fontsize=10)
        plt.ylabel('Number of Pixels', fontsize=10)
        # Set title and show
        plt.show()


    def localt(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)
        image = color.rgb2gray(image)
        # Set the block size to 35
        image=image*100
        block_size = 35
        # Obtain the optimal local thresholding
        alocal_thresh = threshold_local(image, block_size, offset=10)
        binary_local=image
        binary_local=np.where(binary_local>alocal_thresh,255,0)
        show_image(binary_local, 'Local thresholding')


    def others(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        #image = cv2.imread(image1)
        image = mpimg.imread(image1)
        image = color.rgb2gray(image)
        # Set the block size to 35
        image=image*100
        # Use the try all method on the resulting grayscale image
        fig, ax = try_all_threshold(image, verbose=False, figsize=(20,8))
        # Show the resulting plots
        plt.show()

    def edge(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Make the image grayscale
        image_gray = color.rgb2gray(image)
        image_gray=image_gray*100
        # Apply edge detection filter
        edge_sobel = sobel(image_gray)

        # Show original and resulting image to compare
        plot_comparison(image,edge_sobel,"Edges with Sobel")

    def gauss(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Apply filter
        gaussian_image = gaussian(image, multichannel=True)

        # Show original and resulting image to compare
        plot_comparison(image,gaussian_image,"Reduced sharpness Gaussian")


    def histequal(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Use histogram equalization to improve the contrast
        image_eq =  exposure.equalize_hist(image)

        # Show original and resulting image to compare
        plot_comparison(image,image_eq,"Histogram Equalization")


    def adaptequal(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Use histogram equalization to improve the contrast
        image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)

        # Show original and resulting image to compare
        plot_comparison(image,image_adapteq,"Adaptive Equalized")

    def erosion(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        image_gray = color.rgb2gray(image)
        image_gray=image_gray*100
        soaps_image_gray=np.where(image_gray>0,1,0)
        eroded_image_shape = morphology.binary_erosion(image_gray)
        plot_comparison(image,eroded_image_shape,"morphology(Erosion)")

    def dilation(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        image_gray = color.rgb2gray(image)
        image_gray=image_gray*100
        soaps_image_gray=np.where(image_gray>0,1,0)
        eroded_image_shape = morphology.binary_dilation(image_gray)
        plot_comparison(image,eroded_image_shape,"morphology(Dilation)")


    def logo(self):
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Initialize the mask
        mask = np.zeros(image.shape[:-1])
        # Set the pixels where the logo is to 1
        mask[210:272, 360:425] = 1
        # Apply inpainting to remove the logo
        image_logo_removed = inpaint.inpaint_biharmonic(image,
                                          mask,
                                          multichannel=True)
        # Show the original and logo removed images
        plot_comparison(image,image_logo_removed,'Image with logo removed')


    def noisy(self):
        from skimage.util import random_noise
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        noisy_image = random_noise(image)
        # Show the original and logo removed images
        plot_comparison(image,noisy_image,'Image with random noise')

    def tvnoise(self):
        from skimage.restoration import denoise_tv_chambolle
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Apply total variation filter denoising
        denoised_image = denoise_tv_chambolle(image,
        weight=0.1,
        multichannel=True)

        # Show the original and logo removed images
        plot_comparison(image,denoised_image,'Image with Denoised')

    def bilateralnoise(self):
        from skimage.restoration import denoise_bilateral
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Apply bilateral filter denoising
        denoised_image = denoise_bilateral(image, multichannel=True)

        # Show the original and logo removed images
        plot_comparison(image,denoised_image,'Image with Denoised')


    def slicsegmentation(self):
        # Import the slic function from segmentation module
        from skimage.segmentation import slic

        # Import the label2rgb function from color module
        from  skimage.color import label2rgb
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Obtain the segmentation with 400 regions
        segments = slic(image, n_segments= 400)

        # Put segments on top of original image to compare
        segmented_image = label2rgb(segments, image, kind='avg')


        # Show the original and logo removed images
        plot_comparison(image,segmented_image,'Segmented Image with 400 superpixels')


    def showcontour(self):
        from skimage import measure

        # Import the label2rgb function from color module
        from  skimage.color import label2rgb
        image1=""
        file="imageprocessing.csv"
        with open(file, 'r',) as file:
            reader = csv.reader(file, delimiter = '\t')
            for row in reader:
                image1=row[0]
                break
        image = mpimg.imread(image1)
        # Make the image grayscale
        image = color.rgb2gray(image)
        image=image*100
        # Obtain the optimal thresh value of the image
        thresh = threshold_otsu(image)
        binary_global=image
        # Apply thresholding and obtain binary image
        binary_global=np.where(binary_global>thresh,255,0)
        # Find the contours with a constant level value of 0.8
        contours = measure.find_contours(binary_global, 0.8)

        # Shows the image with contours found
        s={len(contours) - 4}
        show_image_contour(binary_global, contours,s)



def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_comparison(original, filtered, title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True,sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis('off')
    plt.show()

def show_image_contour(image, contours,s):

    fig, ax = plt.subplots()
    ax.text(0,0,s)
    ax.imshow(image, cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# This defines the Python GUI backend to use for matplotlib
matplotlib.use('TkAgg')


root = Root()

# Create a tkinter button at the bottom of the window and link it with the updateGraph function
#tk.Button(root,text="Update",command=show_image).grid(row=1, column=0)

root.mainloop()
