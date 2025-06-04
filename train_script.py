import agent as a
import matplotlib.pyplot as plt
import math
import random

def get_lr(epoch, initial_lr=0.001, warmup_epochs=50):
    if epoch < warmup_epochs:
        # Постепенно увеличиваем lr от 0 до initial_lr за warmup_epochs эпох
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # После warmup — обычный decay (можно медленный)
        return initial_lr


initial_lr = 0.01
epocheNum = 1
warmup_epochs = 50

plt.ion()
errorslst = []
epochelst = []
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlabel('epoch')
ax.set_ylabel('sqError')
ax.set_title('Training Error per Epoch')
ax.grid()


for epoche in range(0,epocheNum):
    epocheError = 0
    learningRate = get_lr(epoche, initial_lr, warmup_epochs)
    print(f"Epoche : {epoche} : LR = {learningRate}")
    
    for i in range(0,10):
        pngIndex = int(random.uniform(0,499))
        Errors = a.learn("model-1", "learning_dataset", i, learningRate, pngIndex)
        print(f"        Learning number {i} : idx {pngIndex}")
        epocheError += math.pow(Errors.matrix[i][0],2)
    errorslst.append(epocheError)
    epochelst.append(epoche)
    line.set_data(epochelst, errorslst)
    ax.set_xlim(0, max(epochelst) + 1)
    ax.set_ylim(0, max(errorslst) + 1)

    plt.pause(0.001)
    print(f"     Error = {epocheError}")




input()
plt.ioff()



