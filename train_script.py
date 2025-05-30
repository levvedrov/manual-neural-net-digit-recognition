import agent as a
import matplotlib.pyplot as plt
import math

learningRate = 0.005

epocheError = 0
for i in range(0,10):
    Errors = a.learn("model-1", "learning_dataset", i, learningRate)
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
for epoche in range(0,10):
    epocheError = 0
    print(f"Epoche : {epoche}")
    for i in range(0,10):
        Errors = a.learn("model-1", "learning_dataset", i, learningRate)
        epocheError += math.pow(Errors.matrix[i][0],2)
    errorslst.append(epocheError)
    epochelst.append(epoche)
    plt.draw()
    plt.pause(0.001)
    ax.plot(epochelst, errorslst, color='red')
    print(f"     Error = {epocheError}")
plt.show()
print(f"error improve: {er0-epocheError}")



