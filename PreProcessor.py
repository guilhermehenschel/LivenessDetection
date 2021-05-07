import cv2
import os
from mtcnn.mtcnn import MTCNN

def recenter (x1, y1, width, height):
    xc = x1 + width/2
    yc = y1 + height/2

    if width > height:
        new_height = width
        new_width = width
    else:
        new_width = height
        new_height = height

    xl = int(xc - new_width/2)
    xr = int(xc + new_width/2)
    yt = int(yc - new_height/2)
    yb = int(yc + new_height/2)

    return xl, xr, yt, yb


def get_faces(data, result_list):
    faces = []

    # para cada conjunto de coordenada de faces, retorna a face
    for i in range(len(result_list)):
        # coordenadas
        x1, y1, width, height = result_list[i]['box']
        xl, xr, yt, yb = recenter(x1, y1, width, height)
        # x2, y2 = x1 + width, y1 + height

        # adiciona a face na lista de retorno
        faces.append(data[yt:yb, xl:xr])

    return faces

def pre_processamento(pixels):
    detector = MTCNN()
    faces_info = detector.detect_faces(pixels)
    face = get_faces(pixels, faces_info)[0]

    return face

def prepare_base():
    lista = []
    n = 1000
    X = []

    path = r"E:/NUAA/NUAA/raw"

    dir_list=[]
    dir_list.append(os.listdir(path))
    imagens = []

    labels = ["live","spoof"]

    for super_dir in dir_list[0]:
        tmp_path = path+"/"+str(super_dir)
        mid_dir_list = os.listdir(tmp_path)
        for dir in mid_dir_list:
            tmp_path2 = path+"/"+str(super_dir)+"/"+dir
            files = os.listdir(tmp_path2)
            for file in files:
                path2 = tmp_path2 + "/" + file
                try:
                    pixels = cv2.imread(path2, 1)
                    pixelsn = pre_processamento(pixels)
                    new_path = 'Data/'+super_dir+'/'+file
                    cv2.imwrite(new_path, pixelsn)
                except:
                    print(f"{path2} failed to open!")

if __name__ == '__main__':
    prepare_base()

