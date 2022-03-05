from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


def show_40_distinct_people(images, unique_ids):
    """
    Show 40 distinc people
    """
    
    # Creating 4X10 subplots in  18x9 figure size
    fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))

    # For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr = axarr.flatten()

    # iterating over user ids
    for unique_id in unique_ids:
        image_index = unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))

    # showm results
    plt.suptitle("There are 40 distinct people in the dataset")
    plt.show()


def show_10_faces_of_n_subject(images, subject_ids):
    """
    Show 10 faces of n subject ids
    """
    cols = 10  # each subject has 10 distinct face images
    rows = (len(subject_ids)*10)/cols
    rows = int(rows)

    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 9))
q
    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index = subject_id*10 + j
            axarr[i, j].imshow(images[image_index], cmap="gray")
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            axarr[i, j].set_title("face id:{}".format(subject_id))

    # showm results
    plt.show()


def main():
    """
    Main program
    """

    # load the dataset
    data = np.load('data/olivetti_faces.npy')
    target = np.load('data/olivetti_faces_target.npy')

    # plot samples
    show_40_distinct_people(data, np.unique(target))
    show_10_faces_of_n_subject(images=data, subject_ids=[0,5, 21, 24, 36])
  
    # We reshape images for machine learnig  model
    X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    print("X shape:", X.shape)

    # split train and validation dataset
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
    print("X_train shape:", X_train.shape)
    print("y_train shape:{}".format(y_train.shape))

    # define number of components
    number_of_eigenfaces = 90

    # apply PCA
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)
    
    # show mean face
    plt.imshow(pca.mean_.reshape((64, 64)), cmap="gray")
    plt.show()

    # create the eignen faces
    eigen_faces = pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))

    # show results
    cols = 10
    rows = int(number_of_eigenfaces/cols)
    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    axarr = axarr.flatten()
    for i in range(number_of_eigenfaces):
        axarr[i].imshow(eigen_faces[i], cmap="gray")
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
        axarr[i].set_title("eigen id:{}".format(i))
    plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
    plt.show()

    # get pca for train and test dataset
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # use a C-Support Vector Classification to classify the data
    clf = SVC()
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print("accuracy score:{:.2f}".format(accuracy_score(y_test, y_pred)))

    # show results
    plt.figure(1, figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred))
    plt.show()


if __name__ == "__main__":
    main()
