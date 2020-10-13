# Imports
import tkinter as tk
from PIL import ImageGrab, ImageShow, Image
import numpy as np
import pandas as pd
# import tensorflow as tf
# print(tf.__version__)
import DataAnalysis as DA

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

SMALLER_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALLER_SIZE)  # fontsize of the figure title


# Class Definitions
class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.canvas_width = 280
        self.canvas_height = 280
        self.cmapcnn = 'viridis'

        self.Analyzer = DA.DataAnalyzer()
        self.runTSNE = False

        self.canvasframe = tk.Frame(self)
        self.cbuttons_sub = tk.Frame(self.canvasframe)
        self.graphsframe = tk.Frame(self)
        self.visualizeframe = tk.Frame(self)

        self.canvasframe.pack(side="top", fill="both", expand=True)
        # self.visualizeframe.pack(side="left", fill="both", expand=True)
        self.graphsframe.pack(side="left", fill="both", expand=True)

        self.window = tk.Canvas(self.canvasframe, bg='#000000', width=self.canvas_width, height=self.canvas_height)
        self.Button1 = tk.Button(self.cbuttons_sub, text="Update", command=self.updateallgraphs)
        self.ButtonDelete = tk.Button(self.cbuttons_sub, text="Clear", command=self.deleteall)
        self.ButtonToggleTSNE = tk.Button(self.cbuttons_sub, text="Toggle TSNE", command=self.toggleTSNE)
        self.ButtonOpenCNN = tk.Button(self.cbuttons_sub, text='CNN Activations', command=self.openCNNwindow)

        self.toplevelcnnWindow = tk.Toplevel(self.parent)
        self.cnnImgPerRow = 8
        self.cnnwin_cnnfig = plt.Figure()
        self.cnnwin_cnnfig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.97, wspace=0, hspace=0)
        self.cnnwin_cnnax = self.cnnwin_cnnfig.add_subplot(111)
        self.cnnwin_cnnwidg = FigureCanvasTkAgg(self.cnnwin_cnnfig, self.toplevelcnnWindow)
        self.cnnwin_cnnwidg.draw()
        self.cnnwin_cnnwidg.get_tk_widget().pack()

        self.cnnwin_cnnfig2 = plt.Figure()
        self.cnnwin_cnnfig2.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.9, wspace=0, hspace=0)
        self.cnnwin_cnnax2 = self.cnnwin_cnnfig2.add_subplot(111)
        self.cnnwin_cnnwidg2 = FigureCanvasTkAgg(self.cnnwin_cnnfig2, self.toplevelcnnWindow)
        self.cnnwin_cnnwidg2.draw()
        self.cnnwin_cnnwidg2.get_tk_widget().pack()

        self.toplevelcnnWindow.destroy()

        self.tsneValue = tk.StringVar()
        self.tsneValue.set("Run tSNE: " + str(self.runTSNE))
        self.labelTSNE = tk.Label(self.cbuttons_sub, textvariable=self.tsneValue)
        # self.dnn = tf.keras.models.load_model('./mnist_dnn.h5')

        self.dnnfig = plt.Figure(figsize=(7,7), dpi=100)
        self.dnnfig.subplots_adjust(left=0.1, bottom=0.05, right=0.98, top=0.95, wspace=0.2, hspace=0.3)
        self.axtsne = self.dnnfig.add_subplot(221)
        self.axpca = self.dnnfig.add_subplot(223)

        self.axdnn = self.dnnfig.add_subplot(322)
        self.axcnn = self.dnnfig.add_subplot(324)
        self.axknn = self.dnnfig.add_subplot(326)


        self.axdnn.set_title('ANN Prediction')
        self.axcnn.set_title('CNN Prediction')
        self.axtsne.set_title('t-SNE')
        self.axpca.set_title('PCA')
        self.axknn.set_title('KNN Prediction')

        self.bardnn = FigureCanvasTkAgg(self.dnnfig, self.graphsframe)
        # self.vizfig = plt.Figure(figsize=(2.8,6), dpi=100)

        # self.scattertsne = FigureCanvasTkAgg(self.vizfig, self.visualizeframe)
        # self.scatterpca = FigureCanvasTkAgg(self.vizfig, self.visualizeframe)

        self.numpyfig = plt.Figure(figsize=(2.8,2.8), dpi=100)
        self.axnumpyimg = self.numpyfig.add_subplot(111)
        self.axnumpyimg.set_title('Downsampled Image')
        self.numpyimg = FigureCanvasTkAgg(self.numpyfig, self.canvasframe)
        self.numpyimg.draw()
        self.numpyimg.get_tk_widget().pack(side=tk.RIGHT)

        # Window Packs
        self.window.pack(side=tk.LEFT)
        self.cbuttons_sub.pack(side=tk.RIGHT)
        self.ButtonOpenCNN.pack(side="bottom")
        self.labelTSNE.pack(side="bottom")
        self.ButtonToggleTSNE.pack(side="bottom", fill="both")
        self.ButtonDelete.pack(side="bottom", fill="both")
        self.Button1.pack(side="bottom", fill="both")
        self.bardnn.draw()
        self.bardnn.get_tk_widget().pack(fill="both")
        # self.scattertsne.draw()
        # self.scattertsne.get_tk_widget().pack(side=tk.LEFT)
        # self.scatterpca.draw()
        # self.scatterpca.get_tk_widget().pack(side=tk.BOTTOM)
        # Keybinds
        self.window.bind("<B1-Motion>", self.paint)


    def paint(self, event):
        brushsize = 10
        python_green = '#FFFFFF'
        x1, y1 = (event.x - brushsize), (event.y - brushsize)
        x2, y2 = (event.x + brushsize), (event.y + brushsize)
        self.window.create_oval(x1, y1, x2, y2, fill=python_green, outline="")

    def imagegrabber(self):
        x2 = self.parent.winfo_rootx()+self.window.winfo_x()
        y2 = self.parent.winfo_rooty()+self.window.winfo_y()
        x1 = x2+self.canvas_width
        y1 = y2+self.canvas_height
        # print(x2, y2, w.winfo_width(), w.winfo_height())
        img = ImageGrab.grab(bbox=(x2, y2, x1, y1)).convert("L")
        # img.show()
        img = img.resize((28,28))
        imgnp = np.array(img)

        # np.reshape(imgnp, newshape=(28,28))
        # print(imgnp.shape)

        # lotpred = [i for i, _ in enumerate(pred[0])]

        # plt.imshow(imgnp, cmap=plt.get_cmap('gray'))
        # plt.show()
        # print(imgnp)
        return imgnp

    def toggleTSNE(self):
        self.runTSNE = not self.runTSNE
        self.tsneValue.set("Run tSNE: " + str(self.runTSNE))
        self.parent.update_idletasks()

    def updateallgraphs(self, event = None):
        print("Updating!")
        imgnp = self.imagegrabber()

        # pred = [[0, 0, 0.1, 0.7, 0.05, 0.05, 0, 0,0,0]]
        pred_dnn = self.Analyzer.getDNN(imgnp)
        pred_cnn = self.Analyzer.getCNN(imgnp)
        # print(pred_dnn)

        self.axnumpyimg.cla()
        self.axnumpyimg.set_title('Downsampled Image')
        self.axnumpyimg.imshow(imgnp, cmap='gray')
        self.numpyimg.draw()

        self.axdnn.cla()
        self.axcnn.cla()

        self.axdnn.set_title('ANN Prediction')
        self.axcnn.set_title('CNN Prediction')
        self.axtsne.set_title('t-SNE')
        self.axpca.set_title('PCA')
        self.axknn.set_title('KNN Prediction')

        self.axdnn.bar(range(len(pred_dnn)), pred_dnn)
        self.axcnn.bar(range(len(pred_cnn)), pred_cnn)

        loc = matplotlib.ticker.MultipleLocator(base=1.0)
        self.axdnn.xaxis.set_major_locator(loc)
        self.axdnn.set_ylim([0,1])
        self.axcnn.xaxis.set_major_locator(loc)
        self.axcnn.set_ylim([0,1])
        self.bardnn.draw()

        if self.toplevelcnnWindow.winfo_exists():
            self.cnnwin_cnnax.cla()
            self.cnnwin_cnnax2.cla()

            self.cnnwin_cnnax.set_title('First Convolution Layer Activations', fontsize=12)
            self.cnnwin_cnnax2.set_title('Second Convolution Layer Activations', fontsize=12)
            self.cnnwin_cnnax.get_xaxis().set_visible(False)
            self.cnnwin_cnnax.get_yaxis().set_visible(False)
            self.cnnwin_cnnax2.get_xaxis().set_visible(False)
            self.cnnwin_cnnax2.get_yaxis().set_visible(False)

            display_grid, scale = self.Analyzer.getIntermediate0(self.cnnImgPerRow)
            display_grid2, scale2 = self.Analyzer.getIntermediate1(self.cnnImgPerRow)

            self.cnnwin_cnnax.imshow(display_grid, cmap=self.cmapcnn)
            self.cnnwin_cnnax2.imshow(display_grid2, cmap=self.cmapcnn)
            self.cnnwin_cnnwidg.draw()
            self.cnnwin_cnnwidg2.draw()

        if self.runTSNE:
            self.axtsne.cla()
            tsne_data, tsne_img = self.Analyzer.getTSNE(imgnp)
            tsnedf = pd.DataFrame(tsne_data, columns=("Dim 1", "Dim 2", "label"))
            for i, dff in tsnedf.groupby("label"):
                self.axtsne.scatter(dff["Dim 1"], dff["Dim 2"], label=int(i))
            self.axtsne.legend()
            self.axtsne.scatter(tsne_img[0], tsne_img[1], s=75, color='black')
            self.axtsne.scatter(tsne_img[0], tsne_img[1], s=25, color="C"+str(np.argmax(pred_cnn)))
            self.bardnn.draw()


    def deleteall(self):
        self.window.delete("all")



    def openCNNwindow(self):
        if not self.toplevelcnnWindow.winfo_exists():
            self.toplevelcnnWindow = tk.Toplevel(self.parent)
            self.toplevelcnnWindow.title("CNN WINDOW")
            x2 = self.parent.winfo_rootx()
            y2 = self.parent.winfo_rooty()
            self.toplevelcnnWindow.geometry("800x1000+"+str(x2+self.parent.winfo_width()+10)+"+"+str(y2))

            if self.Analyzer.cnn_interpred0 is not None:
                self.cnnwin_cnnwidg = FigureCanvasTkAgg(self.cnnwin_cnnfig, self.toplevelcnnWindow)
                self.cnnwin_cnnwidg2 = FigureCanvasTkAgg(self.cnnwin_cnnfig2, self.toplevelcnnWindow)

                display_grid, scale = self.Analyzer.getIntermediate0(self.cnnImgPerRow)
                display_grid2, scale2 = self.Analyzer.getIntermediate1(self.cnnImgPerRow)

                self.cnnwin_cnnfig.set_size_inches(scale*display_grid.shape[1], scale*display_grid.shape[0])
                self.cnnwin_cnnfig2.set_size_inches(scale2*display_grid2.shape[1], scale2*display_grid2.shape[0])

                self.cnnwin_cnnax.imshow(display_grid, cmap=self.cmapcnn)
                self.cnnwin_cnnax2.imshow(display_grid2, cmap=self.cmapcnn)

                self.cnnwin_cnnax.set_title('First Convolution Layer Activations', fontsize=12)
                self.cnnwin_cnnax2.set_title('Second Convolution Layer Activations', fontsize=12)
                self.cnnwin_cnnax.get_xaxis().set_visible(False)
                self.cnnwin_cnnax.get_yaxis().set_visible(False)
                self.cnnwin_cnnax2.get_xaxis().set_visible(False)
                self.cnnwin_cnnax2.get_yaxis().set_visible(False)


                self.cnnwin_cnnwidg.draw()
                self.cnnwin_cnnwidg.get_tk_widget().pack(fill="both")
                self.cnnwin_cnnwidg2.draw()
                self.cnnwin_cnnwidg2.get_tk_widget().pack(fill="both")







        else:
            self.toplevelcnnWindow.destroy()
            # print("Closing CNN Window")


def main():
    root = tk.Tk()
    MainApp(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

if __name__=="__main__":
    main()
