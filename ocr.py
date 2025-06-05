from fast_plate_ocr import ONNXPlateRecognizer

m = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')
print(m.run('plates/plate_3.jpg'))