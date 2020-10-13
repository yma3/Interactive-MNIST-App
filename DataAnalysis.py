import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class DataAnalyzer():
    '''
    This class holds all the data pertaining to the TKinter application

    DNN - Pretrained Deep Neural Network on MNIST loaded through KERAS
    CNN - Pretrained Convolutional Neural Network
    KNN - K-Nearest Neighbor Classifier

    PCA - contains transformation matrix acquired through PCA
    LDA?
    TSVD - truncated SVD
    t-SNE - algo
    '''

    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.mnist.load_data()
        self.X_train, self.X_test = self.X_train/255., self.X_test/255.

        self.dnn = tf.keras.models.load_model('./models/mnist_dnn.h5')
        self.cnn = tf.keras.models.load_model('./models/mnist_cnn.h5')
        self.cnn_layer0 = tf.keras.models.Model(inputs=self.cnn.input, outputs=self.cnn.get_layer('conv2d').output)
        self.cnn_layer1 = tf.keras.models.Model(inputs=self.cnn.input, outputs=self.cnn.get_layer('conv2d_1').output)
        self.cnn_interpred0 = None
        self.cnn_interpred1 = None

        self.modelTSNE = TSNE(n_components=2, random_state=0, verbose=1)
        self.tsnedf = None
        self.TSNE_NSAMPLES = 200
        self.tsneX = self.X_train.flatten().reshape(self.X_train.shape[0],-1)


    def getDNN(self, inputimage):
        return self.dnn.predict(np.expand_dims(inputimage, 0))[0]

    def getCNN(self, inputimage):
        img = np.expand_dims(inputimage, -1)
        img = np.expand_dims(img, 0)
        # print(img.shape)
        self.cnn_interpred0 = self.cnn_layer0.predict(img)
        self.cnn_interpred1 = self.cnn_layer1.predict(img)
        return self.cnn.predict(img)[0]

    def getKNN(self):
        return "knn prediction"

    def getPCA(self):
        return "PCA vectors"

    def getTSNE(self, image):
        # Flatten Image
        img = image.flatten().reshape(1,-1)
        # print(img.shape)
        train = self.tsneX[:self.TSNE_NSAMPLES,:]
        train_aug = np.vstack((train, img))
        labels = self.y_train[:self.TSNE_NSAMPLES]
        print(train.shape)
        tsne_data = self.modelTSNE.fit_transform(train_aug)
        print(tsne_data.shape)
        tsne_img = tsne_data[-1,:]
        # print(tsne_data)
        # print(tsne_img)
        tsne_data = np.vstack((tsne_data[:self.TSNE_NSAMPLES].T, labels)).T
        return tsne_data, tsne_img

    def getmodelsummary(self, var):
        var.summary()

    def getIntermediate0(self, images_per_row=16):
        images_per_row = images_per_row
        interpred = self.cnn_interpred0

        n_features = interpred.shape[-1] # [1, 26, 26, >32<]
        size = interpred.shape[1] # [1, >26<, 26, 32]
        n_cols = n_features//images_per_row
        display_grid = np.zeros((size*n_cols, images_per_row*size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_img = interpred[0, :, :, col*images_per_row+row]
                # Processing if needed
                display_grid[col*size : (col+1) * size,
                             row*size : (row+1) * size] = channel_img
        scale = 1./size
        return display_grid, scale

    def getIntermediate1(self, images_per_row=16):
        images_per_row = images_per_row
        interpred = self.cnn_interpred1

        n_features = interpred.shape[-1] # [1, 26, 26, >32<]
        size = interpred.shape[1] # [1, >26<, 26, 32]
        n_cols = n_features//images_per_row
        display_grid = np.zeros((size*n_cols, images_per_row*size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_img = interpred[0, :, :, col*images_per_row+row]
                # Processing if needed
                display_grid[col*size : (col+1) * size,
                             row*size : (row+1) * size] = channel_img
        scale = 1./size
        return display_grid, scale

if __name__ == '__main__':
    colors = ['C0', 'C1', 'C2','C3','C4','C5','C6','C7','C8','C9']

    Analyzer = DataAnalyzer()
    testimg = Analyzer.X_test[0]
    '''
    tsne_data, tsne_img = Analyzer.getTSNE(Analyzer.X_test[0])
    tsnedf = pd.DataFrame(tsne_data, columns=("Dim 1", "Dim 2", "label"))
    val = Analyzer.y_test[0]
    print(val)
    fig, ax = plt.subplots()

    for i, dff in tsnedf.groupby("label"):
        ax.scatter(dff["Dim 1"], dff["Dim 2"], label=int(i))

    # scatter = ax.scatter(tsne_data[:,0], tsne_data[:,1], c=[colors[int(i)] for i in tsne_data[:,2]])
    # l1 = ax.legend(*scatter.legend_elements())
    ax.legend()

    ax.scatter(tsne_img[0], tsne_img[1], s=75, color='black')
    ax.scatter(tsne_img[0], tsne_img[1], s=25, color="C"+str(val))

    plt.show()'''
    Analyzer.getmodelsummary(Analyzer.cnn)
    _ = Analyzer.getCNN(testimg)

    display_grid, scale = Analyzer.getIntermediate0(8)
    display_grid2, scale2 = Analyzer.getIntermediate1(16)

    fig = plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plt.grid(False)
    ax1.imshow(display_grid)
    ax2.imshow(display_grid2)
    plt.show()
    print("Done")

    # Yes
