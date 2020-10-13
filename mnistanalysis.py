import tkinter
from PIL import ImageGrab, ImageShow, Image
import numpy as np
import tensorflow as tf
print(tf.__version__)

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt



canvas_width = 280
canvas_height = 280


# IMPORT ALL KERAS MODELS

mnistdnn = tf.keras.models.load_model('./mnist_dnn.h5')

# ========


def paint( event ):
    brushsize = 8
    python_green = '#FFFFFF'
    x1, y1 = (event.x - brushsize), (event.y - brushsize)
    x2, y2 = (event.x + brushsize), (event.y + brushsize)
    w.create_oval(x1, y1, x2, y2, fill=python_green, outline="")

def imagegrabber():
    x2 = root.winfo_rootx()+w.winfo_x()
    y2 = root.winfo_rooty()+w.winfo_y()
    x1 = x2+canvas_width
    y1 = y2+canvas_height
    # print(x2, y2, w.winfo_width(), w.winfo_height())
    img = ImageGrab.grab(bbox=(x2, y2, x1, y1)).convert("L")
    # img.show()
    img = img.resize((28,28))
    imgnp = np.array(img)
    # np.reshape(imgnp, newshape=(28,28))
    # print(imgnp.shape)
    pred = mnistdnn.predict(np.expand_dims(imgnp, 0))
    print(pred)
    # lotpred = [i for i, _ in enumerate(pred[0])]
    ax1.cla()
    ax1.bar(range(len(pred[0])), pred[0])
    bar1.draw()
    # plt.imshow(imgnp, cmap=plt.get_cmap('gray'))
    # plt.show()
    # print(imgnp)


def updateallgraphs(event = None):
    print("Updating!")
    imagegrabber()
    # root.after(100, updateallgraphs)

def deleteall():
    w.delete("all")


root = tkinter.Tk()
w = tkinter.Canvas(root, bg='#000000', width=canvas_width, height=canvas_height)
Button = tkinter.Button(root, text="Update", command=updateallgraphs)
Button.pack(side=tkinter.RIGHT)
ButtonDelete = tkinter.Button(root, text="Clear", command=deleteall)
ButtonDelete.pack(side=tkinter.RIGHT)
w.pack()
w.bind("<B1-Motion>", paint)
# w.bind("<ButtonRelease-1>", updateallgraphs)
# root['bg'] = '#FFFFFF'

# Define Figure
figure1 = plt.Figure(figsize=(4,3), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, root)
bar1.draw()
bar1.get_tk_widget().pack(side=tkinter.RIGHT)

# ===

# root.after(1000, updateallgraphs)
root.mainloop()
