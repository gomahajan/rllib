import matplotlib.pyplot as plt

def plot(*args):
    plt.figure(2)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    for arg in args:
        plt.plot(arg)
    plt.pause(0.001)