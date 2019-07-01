from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

def plotter(path_list, save_to_disk=False):

        fig, ax = plt.subplots()

        pic = ax.imshow(plt.imread(path_list[0]))

        def animate(index):
                pic.set_array(plt.imread(path_list[index]))
                return pic

        ani = animation.FuncAnimation(fig, animate,np.arange(1,len(path_list)),interval=250)
	
	if save_to_disk:
		ani.save('animation.gif', writer='imagemagick', fps=4)

        plt.show()

def plotter_im(ims, save_to_disk=False):

        fig, ax = plt.subplots()

        pic = ax.imshow(ims[0])

        def animate(index):
                pic.set_array(ims[index])
                return pic

        ani = animation.FuncAnimation(fig, animate,np.arange(1,len(ims)),interval=250)
        
	if save_to_disk:
		ani.save('animation_2.gif', writer='imagemagick', fps=4)

        plt.show()

