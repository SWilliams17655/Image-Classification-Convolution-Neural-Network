import cnn
import cv2 as cv2
import matplotlib.pyplot as plt
import keras_preprocessing.image


def image_display(image):
    """Displays either a greyscale or color image using matplot.
    :param image: The image to be displayed.
    """
    if len(image.shape) == 3:
        print(f"Displaying a color image with shape {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if len(image.shape) == 2:
        print(f"Displaying a greyscale image with shape {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    plt.show()

# Check to ensure GPU is being used to accelerate processing.
count = cv2.cuda.getCudaEnabledDeviceCount()
if count == 1:
    print("The system is using GPU support.")
else:
    print("The system is not using GPU support")

#load image.
img1 = cv2.imread("eval.jpg", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
img2 = keras_preprocessing.image.load_img("eval2.jpg", target_size=(128, 128))
image_display(img1)

cnn_mod = cnn.CNN(["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"])
cnn_mod.train_model("Data/train", "Data/validate", 30)
cnn_mod.graph_training_history()
cnn_mod.classify_image(img1)
