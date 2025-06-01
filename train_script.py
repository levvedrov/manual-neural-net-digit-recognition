import agent as a
import matplotlib.pyplot as plt
import math
import os
import random

learningRate = 0.005
epocheNum = 10000

epocheError = 0
for i in range(0,10):
    for pngIndex in range(100):
        Errors = a.learn("model-1", "learning_dataset", i, learningRate, pngIndex)
        epocheError += math.pow(Errors.matrix[i][0],2)
print(f"     Error = {epocheError}")
er0 = epocheError

errorslst = []
epochelst = []
fig, ax = plt.subplots() 
ax.set_xlabel('epoch')
ax.set_ylabel('sqError')
ax.set_title('Training Error per Epoch')
ax.grid()
plt.show(block=False)



for epoche in range(0,epocheNum):
    epocheError = 0
    print(f"Epoche : {epoche}")

    for i in range(0,10):
        pngIndex = int(random.uniform(0,99))
        Errors = a.learn("model-1", "learning_dataset", i, learningRate, pngIndex)
        print(f"Learning number {i} : idx {pngIndex}")
        epocheError += math.pow(Errors.matrix[i][0],2)

    errorslst.append(epocheError)
    epochelst.append(epoche)


    print(f"     Error = {epocheError}")

ax.plot(epochelst, errorslst, color='red')
plt.draw()
plt.pause(0.001)
plt.show()
print(f"error improve: {er0-epocheError}")



