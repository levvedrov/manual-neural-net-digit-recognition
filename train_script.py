import agent as a

for epoche in range(0,5000):
    print(f"Epoche : {epoche}")
    for i in range(0,10):
        print(f"        learing number: {i}")
        a.learn("model-1", "learning_dataset", i, 0.01)
