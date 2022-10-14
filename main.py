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

color_real = [0, 0, 255] # color: BGR
color_calc = [255, 0, 0] # color: BGR
def calc_intersection_of_circles(center_real,radius_real,center_calc,radius_calc,im_shape):
    img1 = np.zeros(im_shape, np.uint8)
    center_real = (int(center_real[0]),int(center_real[1]))
    center_calc = (int(center_calc[0]),int(center_calc[1]))
    cv2.circle(img1, center_real, int(radius_real), color_real, -1)
    img2 = np.zeros(im_shape, np.uint8)
    cv2.circle(img2, center_calc, int(radius_calc), color_calc, -1)
    return img1 + img2


def calc_dist2d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calc_error(img_validation):
    pixels = np.array(img_validation)
    pixels = pixels.reshape(-1, 3)
    pixels = pixels.tolist()
    #print((np.array(color_calc) + np.array(color_real)).tolist())
    true_negative = []
    true_positive = []
    false_negative = []
    false_positive = []
    for p in pixels:
        #print(p)
        
        if p == [0, 0, 0]:
            true_negative.append(p)
        elif p == color_real:
            false_negative.append(p)
        elif p == color_calc:
            false_positive.append(p)
        elif p == (np.array(color_calc) + np.array(color_real)).tolist():
            true_positive.append(p)
    

    data = {
        "TP":            len(true_positive),
        "FP":            len(false_positive),
        "TN":            len(true_negative),
        "FN":            len(false_negative),
        "TPR":           len(true_positive)/(len(true_positive)+len(false_negative)),
        "FPR":           len(false_positive)/(len(false_positive)+len(true_negative)),
        "TNR":           len(true_negative)/(len(false_positive)+len(true_negative)),
        "FNR":           len(false_negative)/(len(true_positive)+len(false_negative)),
        "ACC":          (len(true_positive)+len(true_negative))/(len(true_positive)+len(true_negative)+len(false_positive)+len(false_negative)),
        "sensitivity":   len(true_positive)/(len(true_positive)+len(false_negative)),
        "specificity":   len(true_negative)/(len(false_positive)+len(true_negative)),
        "weighted_acc":  (len(true_positive)/(len(true_positive)+len(false_negative)) + len(true_negative)/(len(false_positive)+len(true_negative)))/2,
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter('output.avi',fourcc, 2, (240,80))

if __name__ == "__main__":
    print("Iniciando testes")
    df_results = pd.DataFrame()
    # carrega o modelo
    model = OpenCV_HistAnalysys_Model()
    # carrega os dados de teste
    dataset = datasets[0]
    file_names = []
    TPs = []
    FPs = []
    TNs = []
    FNs = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    ACCs = []
    sensitivities = []
    specificities = []
    weights_acc = []
    real_x = [] # p_real[1]
    real_y = [] # p_real[0]
    real_radius = []
    frames = 0
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
                cv2.circle(img, (int(pcalc_center[0]),int(pcalc_center[1])), int(pcalc_radius),  color_calc, 1)
                cv2.circle(img, (int(p_real_center[0]),int(p_real_center[1])), int(p_real_radius), color_real, 1)
                # calcula o erro
                img_validation = calc_intersection_of_circles(p_real_center,p_real_radius,pcalc_center,pcalc_radius,img.shape)
                error = calc_error((img_validation))
                numpy_horizontal_concat = np.concatenate((img, img_validation), axis=1)
                # salva os resultados
                file_names.append(file.split('.png')[0])
                TPs.append(error["TP"])
                FPs.append(error["FP"])
                TNs.append(error["TN"])
                FNs.append(error["FN"])
                TPRs.append(error["TPR"])
                FPRs.append(error["FPR"])
                TNRs.append(error["TNR"])
                FNRs.append(error["FNR"])
                ACCs.append(error["ACC"])
                sensitivities.append(error["sensitivity"])
                specificities.append(error["specificity"])
                weights_acc.append(error["weighted_acc"])
                real_x.append(p_real_center[1])
                real_y.append(p_real_center[0])
                real_radius.append(p_real_radius)
                #pprint(error)
                #print("")

                #cv2.imshow("img",numpy_horizontal_concat)
                if(frames<400):
                    out.write(numpy_horizontal_concat)
                frames+=1
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
        os.system("cls")
    df_results['file_name'] = np.array(file_names)
    df_results['TP'] = np.array(TPs)
    df_results['FP'] = np.array(FPs)
    df_results['TN'] = np.array(TNs)
    df_results['FN'] = np.array(FNs)
    df_results['TPR'] = np.array(TPRs)
    df_results['FPR'] = np.array(FPRs)
    df_results['TNR'] = np.array(TNRs)
    df_results['FNR'] = np.array(FNRs)
    df_results['ACC'] = np.array(ACCs)
    df_results['sensitivity'] = np.array(sensitivities)
    df_results['specificity'] = np.array(specificities)
    df_results['weighted_acc'] = np.array(weights_acc)
    df_results['real_x'] = np.array(real_x)
    df_results['real_y'] = np.array(real_y)
    df_results['real_radius'] = np.array(real_radius)
    df_results.to_csv(f"results/{dataset}.csv",index=False)