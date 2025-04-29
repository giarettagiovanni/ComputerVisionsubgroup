import cv2
import numpy as np
import glob
import os

# Definisco le dimensioni della scacchiera
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
square_size= 0.04

# Creo i vettori dove salvare punti 3D e 2D
objpoints = []   # punti 3D nel mondo reale
imgpoints_left = []  # punti 2D sulla camera sinistra
imgpoints_right = [] # punti 2D sulla camera destra

# Definisco le coordinate 3D per la scacchiera
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

# Carico le immagini
images_left = sorted(glob.glob('./calib_images/left_images/*.jpg'))
images_right = sorted(glob.glob('./calib_images/right_images/*.jpg'))

# Assicurati che siano lo stesso numero di immagini
assert len(images_left) == len(images_right)

for img_left_path, img_right_path in zip(images_left, images_right):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

# Ora calibriamo separatamente entrambe le camere
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# Ora eseguiamo la stereo calibrazione
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC  # Fissa i parametri intrinseci ottenuti sopra

criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retStereo, \
cameraMatrix1, distCoeffs1, \
cameraMatrix2, distCoeffs2, \
R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    gray_left.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

# Stampo i risultati
print("Matrice Camera Sinistra (K_left):\n", cameraMatrix1)
print("\nMatrice Camera Destra (K_right):\n", cameraMatrix2)
print("\nDistorsione Camera Sinistra:\n", distCoeffs1)
print("\nDistorsione Camera Destra:\n", distCoeffs2)
print("\nMatrice di Rotazione R (left -> right):\n", R)
print("\nVettore di Traslazione T (left -> right):\n", T)
baseline = np.linalg.norm(T)
print(f"\nBaseline stimata (distanza fra camere): {baseline*100:.2f} cm")
