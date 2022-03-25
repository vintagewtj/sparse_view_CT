# This program demonstrates the mean RMSE and worst-case ROI RMSE metrics
# The phantom images are taken as the ground truth

import numpy as np, sys, os
import pdb

INPUT = sys.argv[1] # INPUT which has both ./ref and ./res - user submission
OUT = sys.argv[2] # OUTPUT

REFERENCE = os.path.join(INPUT, "ref") # Phantom GT
PREDICTION_OUTPUT = os.path.join(INPUT, "res") # user submission wll be available from here

# Ground Truth
# filtered back-projection reconstruction from the 128-view sinogram
# users will try to recreate these. They serve as the input data.
if len(os.listdir(REFERENCE)) == 1 and os.listdir(REFERENCE)[0][-4:] == ".npy":
    phantom_gt_file_name = os.listdir(REFERENCE)[0]
else:
    raise Exception('Organizer, either you have more than one file in your ref directory or it doesn\'t end in .npy')

# User Images
# The goal is to train a network that accepts the FBP128 image (and/or the 128-view sinogram)
# to yield an image that is as close as possible to the corresponding Phantom image.
if len(os.listdir(PREDICTION_OUTPUT)) == 1 and os.listdir(PREDICTION_OUTPUT)[0][-4:] == ".npy":
    prediction_file_name = os.listdir(PREDICTION_OUTPUT)[0]
else:
    raise Exception('You either have more than one file in your submission or it doesn\'t end in .npy')

phantom_gt = np.load(os.path.join(REFERENCE, phantom_gt_file_name))
prediction_phantoms = np.load(os.path.join(PREDICTION_OUTPUT,prediction_file_name))

# get the number of prediction_phantoms and number of pixels in x and y
nim, nx, ny = prediction_phantoms.shape

# mean RMSE computation
diffsquared = (phantom_gt-prediction_phantoms)**2
num_pix = float(nx*ny)

meanrmse  = np.sqrt( ((diffsquared/num_pix).sum(axis=2)).sum(axis=1) ).mean()
print("The mean RSME over %3i images is %8.6f "%(nim,meanrmse))

# worst-case ROI RMSE computation
roisize = 25  #width and height of test ROI in pixels
x0 = 0        #tracks x-coordinate for the worst-case ROI
y0 = 0        #tracks x-coordinate for the worst-case ROI
im0 = 0       #tracks image index for the worst-case ROI

maxerr = -1.
for i in range(nim): # For each image
   print("Searching image %3i"%(i))
   phantom = phantom_gt[i].copy() # GT
   prediction =  prediction_phantoms[i].copy() # Pred
   # These for loops cross every pixel in image (from region of interest)
   for ix in range(nx-roisize):
      for iy in range(ny-roisize):
         roiGT =  phantom[ix:ix+roisize,iy:iy+roisize].copy() # GT
         roiPred =  prediction[ix:ix+roisize,iy:iy+roisize].copy() # Pred
         if roiGT.max()>0.01: #Don't search ROIs in regions where the truth image is zero
            roirmse = np.sqrt( (((roiGT-roiPred)**2)/float(roisize**2)).sum() )
            if roirmse>maxerr:
               maxerr = roirmse
               x0 = ix
               y0 = iy
               im0 = i
print("Worst-case ROI RMSE is %8.6f"%(maxerr))
print("Worst-case ROI location is (%3i,%3i) in image number %3i "%(x0,y0,im0+1))

with open(os.path.join(OUT,"scores.txt"), "w") as results:
   results.write("score_1: {}\n".format(meanrmse))
   results.write("score_2: {}".format(maxerr))