from PIL import Image, ImageFilter, ImageFile
import numpy as np
from MatrixLab import matrix
import os

def pngMatrix(filename):
    with Image.open(filename) as img:
        img = img.convert("L")
        img = img.resize((28,28))
        img = img.filter(ImageFilter.DETAIL)
        BitImg = np.asarray(img).flatten().tolist()

    MatrixLst = []
    for bit in BitImg:
        MatrixLst.append([bit])
    out = matrix(MatrixLst)
    return out

def importModel(foldername):
    try:
        newMatrix = []
        with open(foldername+'\\input_hidden.dat', 'r') as file:
            for line in file:
                tmp = line.split(':')
                for i in range(len(tmp)):
                    tmp[i] = float(tmp[i])
                newMatrix.append(tmp)
            input_hidden = matrix(newMatrix)
            print(f"input_hidden scheme loaded : matrix {input_hidden.height}x{input_hidden.width}")
    except Exception as e:
        print(f"An error has been occured while the import of the input to hidden layer weights: {e}")
    
    try:
        newMatrix = []
        with open(foldername+'\\hidden_output.dat', 'r') as file:
            for line in file:
                tmp = line.split(':')
                for i in range(len(tmp)):
                    tmp[i] = float(tmp[i])
                newMatrix.append(tmp)
            hidden_output = matrix(newMatrix)
            print(f"hidden_output scheme loaded : matrix {hidden_output.height}x{hidden_output.width}")
    except Exception as e:
        print(f"An error has been occured while the import of the hidden to output layer weights: {e}")

    return input_hidden, hidden_output



input_layer = pngMatrix('test.png')
print(f"InputLayer created : matrix {input_layer.height}x{input_layer.width}")
ih, ho = importModel('model-1')

hidden_layer = ih@input_layer
print(f"HiddenLayer created : matrix {hidden_layer.height}x{hidden_layer.width}")

output_layer = ho@hidden_layer
print(f"OutputLayer created : matrix {output_layer.height}x{output_layer.width}")


print()








