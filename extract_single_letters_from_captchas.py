import os.path
import cv2
import glob
import imutils
import numpy as np

from matplotlib import pyplot as plt


CAPTCHA_IMAGE_FOLDER = "samples_5letters_captcha"
OUTPUT_FOLDER = "extracted_letter_5_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Adaptive thresholding
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    # Otsu thresholding
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu thresholding with Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # try to erase the noise with dilation and erosion
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(th, kernel, iterations=1)
    dilation2 = cv2.dilate(th2, kernel, iterations=1)
    dilation3 = cv2.dilate(th3, kernel, iterations=1)

    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion2 = cv2.erode(dilation2, kernel, iterations=1)
    erosion3 = cv2.erode(dilation3, kernel, iterations=1)

    kernel = np.ones((3, 1), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    dilation2 = cv2.dilate(erosion2, kernel, iterations=1)
    dilation3 = cv2.dilate(erosion3, kernel, iterations=1)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []



    # Get the individual letters.
    x, y, w, h = 34, 15, 22, 50
    for j in range(5):
        # get the bounding rect
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(dilation, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(dilation2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(dilation3, (x, y), (x + w, y + h), (0, 255, 0), 2)
        letter_image_regions.append((x, y, w, h))
        x += w

    titles3 = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
    images3 = [gray, dilation, dilation2, dilation3]

    for l in range(4):
        plt.subplot(2, 2, l + 1), plt.imshow(images3[l], 'gray')
        plt.title(titles3[l])
        plt.xticks([]), plt.yticks([])

    plt.title('Contouring')
    plt.show()


    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = dilation2[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
