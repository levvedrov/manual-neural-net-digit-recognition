from PIL import Image, ImageFilter, ImageFile
import numpy as np
from MatrixLab import matrix
from cv2 import VideoCapture, imshow, applyColorMap, COLORMAP_BONE, imwrite
import os
import math


def sigmoidMatrix(data):
    def __sigmoid(val):
        val = 1/(1+math.exp(-val))
        return val
    newMatrix = []
    for irow in range(len(data.matrix)):
        newMatrix.append([__sigmoid(data.matrix[irow][0])])
    
    return matrix(newMatrix)

def calculateError(outputLayer, actuallVal):
    newMatrix = []
    for row in range(len(outputLayer.matrix)):
        if row == actuallVal:
            newMatrix.append([1-outputLayer.matrix[row][0]])
        else:
            newMatrix.append([0-outputLayer.matrix[row][0]])
    return matrix(newMatrix)

def pngMatrix(filename):
    with Image.open(filename) as img:
        img = img.convert("L")
        img = img.resize((28,28))
        # img = img.filter(ImageFilter.DETAIL)
        BitImg = np.asarray(img).flatten().tolist()

    MatrixLst = []
    for bit in BitImg:
        MatrixLst.append([bit])
    out = matrix(MatrixLst)
    return out

def snapMatrix():
    cam = VideoCapture(0)
    ret, frame = cam.read()
    if ret == True:
        try:
                os.mkdir('./tmp/')
        except:
            pass
        applyColorMap(frame, COLORMAP_BONE)
        imshow('a',frame)
        frame.resize(28,28)
        BitImg = np.asarray(frame).flatten().tolist()
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
    except Exception as e:
        print(f"An error has been occured while the import of the hidden to output layer weights: {e}")

    return input_hidden, hidden_output

def backPropagation(initError, w_inputHidden, w_hiddenOutput):
    hidden_output_error = w_hiddenOutput.transpose() @ initError
    input_hidden_error = w_inputHidden.transpose() @ hidden_output_error
    return input_hidden_error, hidden_output_error

def gradientDescent(rate, input_layer, hidden_layer, output_layer, out_error, e_hidden_output, w_input_hidden, w_hidden_output):
    def __oneMatrix(height):
        newMatrix = []
        for raw in range(height):
            newMatrix.append([1])
        return matrix(newMatrix)


    k_out = __oneMatrix(output_layer.height) - output_layer
    k_out = k_out * output_layer
    delta_output = e_hidden_output * k_out


    dw_hidden_output = (delta_output @ hidden_layer.transpose()) * rate



    out_error = w_hidden_output.transpose() @ delta_output


    k_hid = __oneMatrix(hidden_layer.height) - hidden_layer
    k_hid = k_hid * hidden_layer

    delta_hidden = out_error * k_hid
    dw_input_hidden = (delta_hidden @ input_layer.transpose()) * rate
    return dw_input_hidden, dw_hidden_output


    

def saveModel(folder, inputHidden, hiddenOutput):
    try:
        with open("model-1\\input_hidden.dat", 'w') as file:
            for row in range(100): # 100
                for unit in range(784): # 784
                    file.write(f"{inputHidden.matrix[row][unit]}")
                    if unit != 783:
                        file.write(f":")
                if row !=99:
                    file.write(f"\n")

        with open("model-1\\hidden_output.dat", 'w') as file:
            for row in range(10): # 9
                for unit in range(100): # 100
                    file.write(f"{hiddenOutput.matrix[row][unit]}")
                    if unit != 99:
                        file.write(f":")
                if row !=10:
                    file.write(f"\n")
    except Exception as e:
        print(f"{e}")



def learn(model, dataset, num, learningRate):
    input_layer = pngMatrix(f'{dataset}/{num}.png')
    #print(f"InputLayer creaed : matrix {input_layer.height}x{input_layer.width}")
    w_input_hidden, w_hidden_output = importModel(model)

    hidden_layer = sigmoidMatrix(w_input_hidden@input_layer)
    #print(f"HiddenLayer created : matrix {hidden_layer.height}x{hidden_layer.width}")


    output_layer = sigmoidMatrix(w_hidden_output@hidden_layer)
    #print(f"OutputLayer created : matrix {output_layer.height}x{output_layer.width}")
    # output_layer.show()


    out_error = calculateError(output_layer, num) ###
    input_hidden_error, hidden_output_error = backPropagation(out_error, w_input_hidden, w_hidden_output)

    d_input_hidden, d_hidden_output = gradientDescent(learningRate, input_layer, hidden_layer, output_layer, input_hidden_error, out_error, w_input_hidden, w_hidden_output)
    w_input_hidden-=d_input_hidden
    w_hidden_output-=d_hidden_output
    saveModel(model, w_input_hidden, w_hidden_output )


def detectPng(model, dataset, num):
    input_layer = pngMatrix(f'{dataset}/{num}.png')
    w_input_hidden, w_hidden_output = importModel(model)
    hidden_layer = sigmoidMatrix(w_input_hidden@input_layer)
    output_layer = sigmoidMatrix(w_hidden_output@hidden_layer)
    output_layer.show()
    
