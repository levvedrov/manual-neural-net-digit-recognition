import random
from os import mkdir 
try:
    mkdir("./model-1/")
except:
    pass

with open("model-1\\input_hidden.dat", 'w') as file:
    for row in range(100): # 100
        for unit in range(784): # 784
            file.write(f"{random.randint(0,100)/10000000}")
            if unit != 783:
                file.write(f":")
        if row !=99:
            file.write(f"\n")

with open("model-1\\hidden_output.dat", 'w') as file:
    for row in range(10): # 9
        for unit in range(100): # 100
            file.write(f"{random.randint(0,100)/10000000}")
            if unit != 99:
                file.write(f":")
        if row !=10:
            file.write(f"\n")
