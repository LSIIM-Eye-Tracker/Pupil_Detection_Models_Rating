from models.opencv_hist_analysis import OpenCV_HistAnalysys_Model
from datasets_list import datasets
import os
import cv2
import numpy as np
import pandas as pd
import math
import pickle
from numpy.core.numeric import NaN
from pprint import pprint
from tqdm import tqdm




def calc_dist2d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calc_error(p_calc, p_real, iris_dt):
    pcalc_center, pcalc_radius = p_calc
    preal_center, preal_radius = p_real

    data = {
        "center_error": calc_dist2d(pcalc_center, preal_center),
        "radius_error": abs(pcalc_radius - preal_radius)
    }
    return data
def get_eye_data_from_pickle(pickle_file):
    # open a file, where you stored the pickled data
    file = open(pickle_file, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()
    
    lms = data['ldmks']['ldmks_iris_2d']
    
    iris_dt = [lms[4], lms[1], lms[0], lms[6]]
    lms = data['ldmks']['ldmks_pupil_2d']
    # esquerda,baixo,direita,cima
    pupil_dt = [lms[0], lms[1], lms[3], lms[5]]
    p_real_center = ((pupil_dt[3][0] + pupil_dt[1][0]) /
                     2), ((pupil_dt[2][1] + pupil_dt[0][1])/2)
    p_real_radius = (calc_dist2d(
        pupil_dt[0], pupil_dt[2])/2 + calc_dist2d(pupil_dt[1], pupil_dt[3])/2)/2
    return iris_dt,(pupil_dt,p_real_center,p_real_radius)

if __name__ == "__main__":
    print("Iniciando testes")
    df_results = pd.DataFrame()
    # carrega o modelo
    model = OpenCV_HistAnalysys_Model()
    # carrega os dados de teste
    dataset = datasets[0]
    file_names = []
    center_errors = []
    radius_errors = []
    real_x = [] # p_real[1]
    real_y = [] # p_real[0]
    real_radius = []
    test_data_folders = os.listdir(f"datasets/{dataset}")
    for folder in tqdm(test_data_folders):
        files = os.listdir(f"datasets/{dataset}/{folder}")
        for file in tqdm(files):
            if file.endswith(".png"):
                #print(f"Testando {file}")
                # carrega a imagem
                img = cv2.imread(f"datasets/{dataset}/{folder}/{file}")
                # carrega os dados do pickle
                pickle_file = f"datasets/{dataset}/{folder}/{file.split('.png')[0]}.pkl"
                iris_dt,(pupil_dt,p_real_center,p_real_radius) = get_eye_data_from_pickle(pickle_file)
                # calcula o resultado
                pcalc_center, pcalc_radius = model.detect_pupil(img, iris_dt)
                cv2.circle(img, (int(pcalc_center[0]),int(pcalc_center[1])), int(pcalc_radius), (0, 255, 0), 1)
                cv2.circle(img, (int(p_real_center[0]),int(p_real_center[1])), int(p_real_radius), (0, 0, 255), 1)
                # calcula o erro
                error = calc_error((pcalc_center, pcalc_radius),(p_real_center,p_real_radius),iris_dt)
                # salva os resultados
                file_names.append(file.split('.png')[0])
                center_errors.append(error['center_error'])
                radius_errors.append(error['radius_error'])
                real_x.append(p_real_center[1])
                real_y.append(p_real_center[0])
                real_radius.append(p_real_radius)
                pprint(error)
                #print("")

                cv2.imshow("img",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        os.system("cls")
    df_results['file_name'] = np.array(file_names)
    df_results['center_error'] = np.array(center_errors)
    df_results['radius_error'] = np.array(radius_errors)
    df_results['real_x'] = np.array(real_x)
    df_results['real_y'] = np.array(real_y)
    df_results['real_radius'] = np.array(real_radius)
    df_results.to_csv(f"results/{dataset}.csv",index=False)