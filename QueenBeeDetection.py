# -*- coding: UTF-8 -*-

import os
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class QueenBeeDetection:
    def __init__(self):
        #####
        print("-- Start session --\n")
        self.hog_parameters = {
            "orientations": 16,
            "pixels_per_cell": (16, 16),
            "cells_per_block": (4, 4),
        }
        self.text_labels = ["queen", "worker"]

    def __import_data(self, src_folder, rgb_mode):
        #####
        data = []
        labels = []

        for sub_folder in self.text_labels:
            list_img_filename = os.listdir("{}/{}/".format(src_folder, sub_folder))

            for img_filename in list_img_filename:
                labels.append(sub_folder)

                img_in = cv2.imread(
                    "{}/{}/{}".format(src_folder, sub_folder, img_filename)
                )

                if rgb_mode:
                    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
                    img_in[:, :, 0] = cv2.equalizeHist(img_in[:, :, 0])
                else:
                    if len(img_in.shape) == 3:
                        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

                    img_in = cv2.equalizeHist(img_in)

                hog_hist = hog(
                    img_in,
                    orientations=self.hog_parameters["orientations"],
                    pixels_per_cell=self.hog_parameters["pixels_per_cell"],
                    cells_per_block=self.hog_parameters["cells_per_block"],
                    visualize=False,
                    feature_vector=True,
                    multichannel=rgb_mode,
                )

                data.append(hog_hist)

        self.data = np.array(data)
        self.labels = np.array(labels)

    def train_model(self, src_folder, rgb_mode=True, nbr_cpnt=30):
        #####

        # import data and extract hog features
        self.__import_data(src_folder, rgb_mode)

        # dimensionality reduction
        self.data = preprocessing.minmax_scale(self.data, feature_range=(0, 1), axis=1)
        self.data = self.data.astype("float32")
        pca = PCA(n_components=nbr_cpnt)
        self.data = pca.fit_transform(self.data)

        # traning model
        svm = SVC(
            C=1.0,
            kernel="rbf",
            random_state=10,
            decision_function_shape="ovr",
            gamma="auto",
        )
        svm.fit(self.data, self.labels)

        # save models
        pickle.dump(pca, open("pca.csv", "wb"))
        pickle.dump(svm, open("svm.csv", "wb"))

        print("Model training completed.")

    def recognition(self, src_image, rgb_mode=True):
        #####

        # load models
        pca = pickle.load(open("pca.csv", "rb"))
        svm = pickle.load(open("svm.csv", "rb"))

        # read image and extract hog feature vector
        img_in = cv2.imread(src_image)

        if rgb_mode:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
            img_in[:, :, 0] = cv2.equalizeHist(img_in[:, :, 0])
        else:
            if len(img_in.shape) == 3:
                img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

            img_in = cv2.equalizeHist(img_in)

        hog_hist = hog(
            img_in,
            orientations=self.hog_parameters["orientations"],
            pixels_per_cell=self.hog_parameters["pixels_per_cell"],
            cells_per_block=self.hog_parameters["cells_per_block"],
            visualize=False,
            feature_vector=True,
            multichannel=rgb_mode,
        )

        data = np.array([hog_hist])

        # apply dimensionality reduction
        data = preprocessing.minmax_scale(data, feature_range=(0, 1), axis=1)
        data = data.astype("float32")
        data = pca.transform(data)

        # perform recognition
        predicted = svm.predict(data)[0]
        print(
            "From the input image, the predicted class/label -> {}.".format(predicted)
        )

    def __mean_conf_mat(self, conf_mats):
        #####
        conf_mat = np.zeros(conf_mats[0].shape)

        for element in conf_mats:
            for i in range(conf_mats[0].shape[0]):
                for j in range(conf_mats[0].shape[0]):
                    conf_mat[i][j] += element[i][j]

        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]

        for idx in range(len(self.text_labels)):
            for jdx in range(len(self.text_labels)):
                conf_mat[idx][jdx] = "%0.2f" % (conf_mat[idx][jdx] * 100)

        return conf_mat.tolist()

    def __evaluate(self):
        #####
        accuracy_vect = []
        conf_mat_vect = []

        for fold in self.folds:
            self.clf.fit(
                self.data_r[fold["train_index"]], self.labels[fold["train_index"]]
            )

            predicted = self.clf.predict(self.data_r[fold["test_index"]])
            accuracy_vect.append(
                accuracy_score(self.labels[fold["test_index"]], predicted)
            )
            conf_mat_vect.append(
                confusion_matrix(
                    self.labels[fold["test_index"]],
                    predicted,
                    labels=self.text_labels,
                )
            )

        accuracy = "%0.2f" % ((float(sum(accuracy_vect)) / len(accuracy_vect)) * 100)
        conf_mat = self.__mean_conf_mat(np.array(conf_mat_vect))

        return accuracy, conf_mat

    def model_evaluation(self, src_folder, rgb_mode=True):
        #####

        # import data and extract hog features
        self.__import_data(src_folder, rgb_mode)

        # dimensionality reduction
        self.data = preprocessing.minmax_scale(self.data, feature_range=(0, 1), axis=1)
        self.data = self.data.astype("float32")

        # classification
        self.clf = SVC(
            C=1.0,
            kernel="rbf",
            random_state=10,
            decision_function_shape="ovr",
            gamma="auto",
        )

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        k_folds = skf.split(self.data, self.labels)

        self.folds = []

        for train_index, test_index in k_folds:
            self.folds.append(
                {"train_index": train_index.tolist(), "test_index": test_index.tolist()}
            )

        n_components_max = min(len(self.data), len(self.data[0]))
        best_accuracy = 0

        for nbr_cpnt in range(10, n_components_max, 5):

            pca = PCA(n_components=nbr_cpnt)
            self.data_r = pca.fit_transform(self.data)

            accuracy, conf_mat = self.__evaluate()

            if float(accuracy) > best_accuracy:
                best_accuracy = float(accuracy)
                best_conf_mat = conf_mat
                best_nbr_cpnt = nbr_cpnt

        # print accuracy
        print(
            "{}% \t<- Evaluation using cross-validation (10-folds).".format(
                best_accuracy
            )
        )

        # print number of principal components
        print(
            "{} \t<- Optimized number of principal components.".format(
                str(best_nbr_cpnt).zfill(3)
            )
        )

        # print confusion matrix
        print("\n\tqueen\tworker")
        print("queen\t{}%\t{}%".format(best_conf_mat[0][0], best_conf_mat[0][1]))
        print("worker\t{}%\t{}%".format(best_conf_mat[1][0], best_conf_mat[1][1]))

    def __del__(self):
        #####
        print("\n-- End session --")