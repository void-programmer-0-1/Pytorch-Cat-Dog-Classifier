


with open("TestData/Cat/5.jpg","rb") as image:
    image_file = image.read()
    image_bytes = bytearray(image_file)
    print(image_bytes)
    