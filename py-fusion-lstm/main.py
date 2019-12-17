
import numpy as np
from all_lstm import iteration


def main():

    test_num=5
    class_num=7

    confusion_matrix=np.zeros((test_num,class_num,class_num),dtype=float)
    for i in range(test_num):
        confusion_matrix[i,:,:]=iteration(i)

    print("finish all train and test")


    print(np.average(confusion_matrix[:, :, :], axis=0))




if __name__ == "__main__":
    main()