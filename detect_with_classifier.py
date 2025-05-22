
#This code is under progress, is highly experimental and might contain bugs.
#Will need further review.
#Adapted from https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/
# import the necessary packages
from vit_model import MyViT  # Assuming you have the ViT model class in a file `vit_model.py`
import json
import torch
import numpy as np
import argparse
import imutils
import time
import cv2
from torchvision import transforms
from imutils.object_detection import non_max_suppression


def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5, minSize=(32, 32)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-p", "--pretrain_path", required=True, help="path to pretrained model")  # Renamed
ap.add_argument("-s", "--size", type=str, default="(32, 32)", help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
ap.add_argument("-l", "--class_labels", type=int, default=-1, help="Pass in as JSON")
args = vars(ap.parse_args())

# Load class names from the provided JSON file
with open(args["class_labels"], 'r') as f:
    class_names = json.load(f)

# initialize variables used for the object detection procedure
WIDTH = 400
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (32, 32)  # Resize to 32x32 for ViT model input

# Load the Vision Transformer (ViT) model
print("[INFO] loading Vision Transformer model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = MyViT(chw=(3, 32, 32), n_patches=7, n_blocks=2, hidden_d=128, n_heads=8, out_d=10)  # Your ViT model
vit_model.load_state_dict(torch.load(args["pretrain_path"]))  # Use args["pretrain_path"]
vit_model.to(device)
vit_model.eval()

# Preprocessing transformation for ViT
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # Resize to match ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standard ViT normalization
])

# load the input image from disk, resize it such that it has the supplied width, and then grab its dimensions
orig = cv2.imread(args["image"])
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

# initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

# initialize two lists, one to hold the ROIs generated from the image pyramid and sliding window,
# and another list used to store the (x, y)-coordinates of where the ROI was in the original image
rois = []
locs = []

# time how long it takes to loop over the image pyramid layers and sliding window locations
start = time.time()

# loop over the image pyramid
for image in pyramid:
    # determine the scale factor between the *original* image dimensions and the *current* layer of the pyramid
    scale = W / float(image.shape[1])
    
    # for each layer of the image pyramid, loop over the sliding window locations
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # scale the (x, y)-coordinates of the ROI with respect to the *original* image dimensions
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        
        # preprocess the ROI to match ViT input size
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = transform(roi)  # Apply the ViT transformation
        roi = roi.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # update our list of ROIs and associated coordinates
        rois.append(roi)
        locs.append((x, y, x + w, y + h))
        
        # check to see if we are visualizing each of the sliding windows in the image pyramid
        if args["visualize"] > 0:
            # clone the original image and then draw a bounding box surrounding the current region
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the visualization and current ROI
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

# show how long it took to loop over the image pyramid layers and sliding window locations
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))

# classify each of the proposal ROIs using ViT and then show how long the classifications took
print("[INFO] classifying ROIs...")
start = time.time()

# Perform predictions using the ViT model
preds = []
for roi in rois:
    with torch.no_grad():
        output = vit_model(roi)  # Get prediction from ViT model
        preds.append(output.cpu().numpy())  # Move output to CPU and append

end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))

# Define a mapping from class indices to human-readable class names

#Add the class labels here
#class_names = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 5: 'class_5', 6: 'class_6', 7: 'class_7', 8: 'class_8', 9: 'class_9'}

# Decode the predictions and initialize a dictionary which maps class labels (keys) to any ROIs associated with that label (values)
labels = {}
for i, pred in enumerate(preds):
    prob = pred.max()  # Get the max probability
    label = pred.argmax()  # Get the predicted class label
    
    # filter out weak detections by ensuring the predicted probability is greater than the minimum probability
    if prob >= args["min_conf"]:
        # grab the bounding box associated with the prediction and convert the coordinates
        box = locs[i]
        # grab the list of predictions for the label and add the bounding box and probability to the list
        L = labels.get(label.item(), [])
        L.append((box, prob))
        labels[label.item()] = L

# loop over the labels for each of detected objects in the image
for label in labels.keys():
    # clone the original image so that we can draw on it
    print("[INFO] showing results for label {}".format(label))
    clone = orig.copy()

    # loop over all bounding boxes for the current label
    for (box, prob) in labels[label]:
        # draw the bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the results *before* applying non-maxima suppression, then clone the image again
    # so we can display the results *after* applying non-maxima suppression
    cv2.imshow("Before", clone)
    clone = orig.copy()

    # extract the bounding boxes and associated prediction probabilities, then apply non-maxima suppression
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    
    # Apply NMS per class label
    boxes = non_max_suppression(boxes, proba)

    # loop over all bounding boxes that were kept after applying non-maxima suppression
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box and label on the image
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label_name = class_names[label]  # Get the human-readable label name
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label_name, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # show the output after applying non-maxima suppression
    cv2.imshow("After", clone)
    cv2.waitKey(0)
