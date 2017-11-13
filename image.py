import numpy as np
import cv2


def update_warp(del_p):

    global warp

    p1 = warp[0][0] - 1
    p2 = warp[1][0]
    p3 = warp[0][1]
    p4 = warp[1][1] - 1
    p5 = warp[0][2]
    p6 = warp[1][2]

    warp[0][0] = 1 + p1 + del_p[0] + p1 * del_p[0] + p3 * del_p[1]
    warp[1][0] = p2 + del_p[1] + p2 * del_p[0] + p4 * del_p[1]
    warp[0][1] = p3 + del_p[2] + p1 * del_p[2] + p3 * del_p[3]
    warp[1][1] = 1 + p4 + del_p[3] + p2 * del_p[2] + p4 * del_p[3]
    warp[0][2] = p5 + del_p[4] + p1 * del_p[4] + p3 * del_p[5]
    warp[1][2] = p6 + del_p[5] + p2 * del_p[4] + p4 * del_p[5]


def initilalize():

    global row, column, warp, gradient, descent, inv_hessian, jacobian

    row, column = np.shape(T)
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[0][0] = 1
    warp[1][1] = 1

    sobelx = cv2.Sobel(T, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(T, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.stack([sobelx, sobely], axis=2)
    gradient = np.expand_dims(gradient, axis=2)
    print gradient[1][0], T[1,0]
    jacobian = np.zeros(shape=(row, column, 2, 6))
    for x in xrange(row):
        for y in xrange(column):
            jacobian[x][y] = np.array(
                [[y, 0, x, 0, 1, 0], [0, y, 0, x, 0, 1]], dtype=np.float32)

    descent = np.matmul(gradient, jacobian)

    hessian = np.matmul(np.transpose(descent, (0, 1, 3, 2)), descent)
    hessian = np.sum(hessian, axis=(0, 1))
    print hessian
    inv_hessian = np.linalg.pinv(hessian)


def iteration(iter):

    val = 1
    i = 1
    while val >= 0.0001 and i <= iter:
        warp_of_I = cv2.warpAffine(I, warp, (column, row))

        error_of_image = warp_of_I - T

        before_del_p = np.matmul(np.transpose(
            descent, (0, 1, 3, 2)), error_of_image.reshape(row, column, 1, 1))
        before_del_p = np.sum(before_del_p, axis=(0, 1))

        del_p = np.dot(inv_hessian, before_del_p)
        del_p = del_p.reshape(6)

        update_warp(del_p)

        val = np.linalg.norm(del_p)
        print i, val
        i += 1


def main(template, image, iter):

    global T, I

    T = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = T.astype(np.float32)
    I = I.astype(np.float32)
    initilalize()
    # iteration(iter)

    # warp_of_I = cv2.warpAffine(image, warp, (column, row))
    # warp_of_I = warp_of_I.astype(np.uint8)
    # cv2.imshow('img', warp_of_I)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':

    template = cv2.imread('T.jpg')
    image = cv2.imread('I.jpg')
    main(template, image, 500)
