from PIL import Image

image = Image.open("Set5/sample_LR.jpg")
biCImage = image.resize((268,268), Image.BICUBIC)
biCImage.show()