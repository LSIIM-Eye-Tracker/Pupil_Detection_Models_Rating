import os
from random import randint
import cv2
import numpy as np
import math
import pickle
from numpy.core.numeric import NaN
from pprint import pprint


class OpenCV_HistAnalysys_Model():
    def convert_to_gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def crop_iris(self, img, iris_borders):
        #print([int(iris_borders[1][1]), int(iris_borders[3][1])], [int(iris_borders[2][0]), int(iris_borders[0][0])])
        img = img[int(iris_borders[1][1]):int(iris_borders[3][1]),
                  int(iris_borders[2][0]):int(iris_borders[0][0])]
        #print(img.shape)
        #cv2.imshow("img", img)
        cv2.waitKey(0)
        return img

    def apply_otsus(self, img):
        ret, o1 = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, o2 = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, o3 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        ret, o4 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
        ret, o5 = cv2.threshold(
            img, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        return [o1, o2, o3, o4, o5]

    def hist_analisys(self, img):
        col_count = np.zeros((img.shape[1]))
        row_count = []
        for i in range(img.shape[0]):
            aux_row = 0
            aux_col = 0
            for j in range(img.shape[1]):
                aux_row = aux_row + img[i][j]/img.shape[1]
                col_count[j] = col_count[j] + img[i][j]/img.shape[0]
            row_count.append(aux_row)
        ##cv2.imshow("img", img)
        for i in range(len(row_count)):
            row_count[i] = (255 - row_count[i])
        for i in range(len(col_count)):
            col_count[i] = (255 - col_count[i])

        row_count = np.array(row_count)

        row_count = row_count - row_count.min()
        media_r = 0
        desvio_r = 0
        media_c = 0
        desvio_c = 0
        if (len(row_count) > 0):
            for i, row in enumerate(row_count):
                media_r += row*i/row_count.sum()

            for i, row in enumerate(row_count):
                # if row>(0.45*row_count.max()):
                if row > 0:
                    desvio_r = media_r - i
                    break

            col_count = col_count - col_count.min()

            for i, col in enumerate(col_count):
                media_c += col*i/col_count.sum()

            for i, col in enumerate(col_count):
                if col > (col_count.max()*0.25):
                    # if col>0:
                    desvio_c = media_c - i
                    break

        if media_c is None or media_c.size <= 0 or media_c == NaN:
            media_c = 0
        if media_r is None or media_r.size <= 0 or media_r == NaN:
            media_r = 0
        if desvio_r is None or desvio_r.size <= 0 or desvio_r < 0:
            desvio_r = 0
        if desvio_c is None or desvio_c.size <= 0 or desvio_c < 0:
            desvio_c = 0
        return row_count, col_count, media_c, media_r, desvio_c, desvio_r

    def analyse_pupil(self, iris_img, iris_borders):
        ##cv2.imshow("iris_img", iris_img)
        #  Testes de filtos pra isolar a pupila
        iris_img = cv2.GaussianBlur(iris_img, (5, 5), 1)

        # #cv2.imshow("left_eye_img_blur",iris_img)
        o1, o2, o3, o4, o5 = self.apply_otsus(iris_img)
        b2 = cv2.GaussianBlur(o5, (5, 5), 1)

        th2 = cv2.adaptiveThreshold(b2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        blurred = cv2.bilateralFilter(b2, 20, 40, 50)

        # #cv2.imshow("b2",b2)

        # #cv2.imshow("th2",th2)

        # #cv2.imshow("blur",blurred)
        #ret, bin_img = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        k = int(25*(len(b2.flatten())-1)/100)
        # print(len(b2.flatten()-1),k)
        idx = np.argpartition(b2.flatten(), k)
        # print(idx)
        # print(b2.flatten()[idx[:k]].mean())
        ret, bin_img = cv2.threshold(
            b2, b2.flatten()[idx[:k]].mean(), 255, cv2.THRESH_BINARY)

        ##cv2.imshow("bin_img 1 ",bin_img)
        row_count, col_count, media_c, media_r, desvio_c, desvio_r = self.hist_analisys(
            bin_img)
        radius = ((desvio_c+desvio_r)/2)
        # print(int(media_c),int(media_r),radius)
        return ((media_c) + iris_borders[2][0], (media_r)+iris_borders[1][1]), radius

    def detect_pupil(self, eye_image, iris_borders):
        gray = self.convert_to_gray_scale(eye_image)
        #cv2.imshow("eye_image", gray)
        cv2.waitKey(0)
        eye_image = self.crop_iris(gray, iris_borders)
        #cv2.imshow("eye_image", eye_image)
        cv2.waitKey(0)
        r = 0
        pupil, r = self.analyse_pupil(eye_image, iris_borders)

        return pupil, r


def calc_dist2d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calc_error(p_calc, p_real):
    pcalc_center, pcalc_radius = p_calc
    preal_center, preal_radius = p_real
    data = {
        "center_error": {
            'x': math.sqrt((((preal_center[1]-pcalc_center[1])/preal_center[1]))**2),
            'y': math.sqrt((((preal_center[0]-pcalc_center[0])/preal_center[0]))**2)
        },
        "radius_error": math.sqrt((((preal_radius-pcalc_radius)/preal_radius))**2)
    }
    return data


if __name__ == "__main__":
    model = OpenCV_HistAnalysys_Model()
    img = cv2.imread(
        os.path.join(".", "datasets", "Cambridge_SynthesEyes_Dataset", "f01", "f01_309_-0.5890_0.3927.png"))

    # open a file, where you stored the pickled data
    file = open(os.path.join(".", "datasets", "Cambridge_SynthesEyes_Dataset",
                "f01", "f01_309_-0.5890_0.3927.pkl"), 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    iris_dt = data['ldmks']['ldmks_iris_2d']
    '''for i, lms in enumerate(iris_dt):
        cv2.putText(img, str(i), (int(lms[0]), int(
            lms[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.circle(img, (int(lms[0]), int(lms[1])), 1, (0, 0, 255), 2)'''

    # direita,cima,esquerda,baixo
    iris_dt = [iris_dt[4], iris_dt[1], iris_dt[0], iris_dt[6]]
    # altuta, largura, canais de cor
    print(img.shape)
    '''for dt in iris_dt:
        print(dt)'''
    pcalc_center, pcalc_radius = model.detect_pupil(img, iris_dt)
    print(pcalc_center, pcalc_radius)

    # pprint(data['ldmks']['ldmks_pupil_2d'])
    lms = data['ldmks']['ldmks_pupil_2d']
    # esquerda,baixo,direita,cima
    pupil_dt = [lms[0], lms[1], lms[3], lms[5]]
    print("centro: ", ((pupil_dt[2][1] + pupil_dt[0][1])/2, (pupil_dt[3][0] + pupil_dt[1][0])/2),
          "| raio: ", (calc_dist2d(pupil_dt[0], pupil_dt[2])/2, calc_dist2d(pupil_dt[1], pupil_dt[3])/2))
    print(len(lms))
    # eu fiz esse loop p ir vendo os pontos aparecendo na tela p saber onde ficava cada um (mas eu ainda posso ter errado)
    esquerda = 0
    direita = 0
    cima = 0
    baixo = 0
    for i in range(1, int(len(lms))-1):
        if (lms[i][0] < lms[esquerda][0]):
            esquerda = i
        if (lms[i][0] > lms[direita][0]):
            direita = i
        if (lms[i][1] < lms[cima][1]):
            cima = i
        if (lms[i][1] > lms[baixo][1]):
            baixo = i
    print(esquerda, baixo, direita, cima)

    '''for i in range(int(len(lms))-1):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(img, (int(lms[i][0]),
                         int(lms[i][1])), 0, color, 2)'''
    img1 = img.copy()

    cv2.circle(img1, (int(pcalc_center[0]), int(pcalc_center[1])),
               int(pcalc_radius), (255, 0, 255), 1)
    img2 = img.copy()
    p_real_center = ((pupil_dt[3][0] + pupil_dt[1][0]) /
                     2), ((pupil_dt[2][1] + pupil_dt[0][1])/2)
    p_real_radius = (calc_dist2d(
        pupil_dt[0], pupil_dt[2])/2 + calc_dist2d(pupil_dt[1], pupil_dt[3])/2)/2
    cv2.circle(img2, (int(p_real_center[0]), int(p_real_center[1])),
               int(p_real_radius), (0, 0, 255), 1)
    pprint(calc_error((pcalc_center, pcalc_radius), (p_real_center, p_real_radius)))
    cv2.imshow("img", img)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
    #cv2.circle(img, (int(lms[0]), int(lms[1])), 1, (0, 0, 255), 2)
