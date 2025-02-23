import cv2
import numpy as np
import os
import glob
import time
from openni import openni2

class KinectCalibrator:
    def __init__(self, openni_path="C:/Program Files/OpenNI2/Redist", chessboard_size=(9, 6), square_size=0.024):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        self.objpoints_rgb = []
        self.objpoints_ir = []
        self.imgpoints_rgb = []
        self.imgpoints_ir = []
        
        self.RGB_dir = "./Fotos_cal_RGB"
        self.IR_dir = "./Fotos_cal_IR"
        os.makedirs(self.RGB_dir, exist_ok=True)
        os.makedirs(self.IR_dir, exist_ok=True)

        self.RGB_imgs = glob.glob(self.RGB_dir + "/*.jpg")
        self.IR_imgs = glob.glob(self.IR_dir + "/*.jpg")
        self.take_pictures = True

        if len(self.RGB_imgs) == len(self.IR_imgs) and len(self.RGB_imgs) >= 50:
            self.take_pictures = False
        
        self.txt_file = "calibration.txt"

        openni2.initialize(openni_path)
        #self.device = openni2.Device.open_any()
        self.rgb_stream = None
        self.ir_stream = None
        self.frame_count = 0

    def start_rgb_stream(self):
        if self.rgb_stream is None:
            mode = openni2.VideoMode(
                pixelFormat=openni2.PIXEL_FORMAT_RGB888,  #formato de 24 bits
                resolutionX=640,
                resolutionY=480,
                fps=60
            )
            self.rgb_stream = self.device.create_color_stream()
            self.rgb_stream.set_video_mode(mode)
            self.rgb_stream.start()

    def start_ir_stream(self):
        if self.ir_stream is None:
            mode = openni2.VideoMode(
                pixelFormat=openni2.PIXEL_FORMAT_GRAY16,  #formato de 16 bits
                resolutionX=640,
                resolutionY=480,
                fps=60
            )
            self.ir_stream = self.device.create_ir_stream()
            self.ir_stream.set_video_mode(mode)
            self.ir_stream.start()
            
    def capture_rgb_frame(self):
        frame = self.rgb_stream.read_frame()
        rgb_data = np.array(frame.get_buffer_as_triplet()).reshape(frame.height, frame.width, 3)
        return cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

    def capture_ir_frame(self):
        frame = self.ir_stream.read_frame()
        ir_data = np.array(frame.get_buffer_as_uint16()).reshape(frame.height, frame.width)
        return cv2.convertScaleAbs(ir_data, alpha=(255.0 / np.max(ir_data)))

    def add_corners_rgb(self, rgb_image):
        gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray_rgb, self.chessboard_size, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_rgb, corners, (11, 11), (-1, -1), criteria)
            self.objpoints_rgb.append(self.objp)
            self.imgpoints_rgb.append(corners)
            return True
        return False

    def add_corners_ir(self, ir_image):
        found, corners = cv2.findChessboardCorners(ir_image, self.chessboard_size, None)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(ir_image, corners, (11, 11), (-1, -1), criteria)
            self.objpoints_ir.append(self.objp)
            self.imgpoints_ir.append(corners)
            return True
        return False

    def calibrate_camera(self, objpoints, imgpoints, image_shape):
        return cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    
    def calibrate_stereo(self, mtx_rgb, dist_rgb, mtx_ir, dist_ir, image_shape):
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints_rgb, self.imgpoints_rgb, self.imgpoints_ir,
            mtx_rgb, dist_rgb, mtx_ir, dist_ir,
            image_shape,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-6),
            flags=flags
        )
        return ret, R, T, E, F
    
    def release_rgb_stream(self):
        if self.rgb_stream is not None:
            self.rgb_stream.stop()
            self.rgb_stream = None
    
    def release_ir_stream(self):
        if self.ir_stream is not None:
            self.ir_stream.stop()
            self.ir_stream = None

    def release(self):
        if self.rgb_stream is not None:
            self.rgb_stream.stop()
            self.rgb_stream = None
        if self.ir_stream is not None:
            self.ir_stream.stop()
            self.ir_stream = None
        openni2.unload()

if __name__ == "__main__":
    calibrator = KinectCalibrator()
    image_shape_rgb = None
    image_shape_ir = None
    state = "RGB"
    #print("Estado RGB: Presiona 'c' para capturar imagen, 'n' para pasar al siguiente estado.")
    
    if calibrator.take_pictures:
        print("Toma de fotos activada.")
        print("Presiona 'c' para capturar imagen, 'n' para finalizar.")
        while calibrator.frame_count < 60:
            calibrator.start_rgb_stream()
            print(f"Frame {calibrator.frame_count}")
            rgb = calibrator.capture_rgb_frame()
            cv2.imshow("RGB cam", rgb)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if calibrator.add_corners_rgb(rgb):
                    print("Esquinas detectadas y agregadas en img RGB.")
                    calibrator.release_rgb_stream()
                    calibrator.start_ir_stream()
                    ir = calibrator.capture_ir_frame()
                    ir = cv2.GaussianBlur(ir, (5, 5), 0)
                    if calibrator.add_corners_ir(ir):
                        print("Esquinas detectadas y agregadas en img ir.")
                        calibrator.release_ir_stream()
                        filename = os.path.join(calibrator.RGB_dir, f"frame_{calibrator.frame_count:05d}.jpg")
                        cv2.imwrite(filename, rgb)
                        filename = os.path.join(calibrator.IR_dir, f"frame_{calibrator.frame_count:05d}.jpg")
                        cv2.imwrite(filename, ir)
                        calibrator.frame_count += 1
                        image_shape_rgb = (rgb.shape[1], rgb.shape[0])
                        image_shape_ir = (ir.shape[1], ir.shape[0])
                    else:
                        print("No se detectaron esquinas en IR.")
                        calibrator.objpoints_rgb.pop()
                        calibrator.imgpoints_rgb.pop()
                        calibrator.release_ir_stream()
                else:
                    print("No se detectaron esquinas en RGB.")
                    calibrator.release_rgb_stream()
                #cv2.destroyAllWindows()
            #time.sleep(0.05)
        cv2.destroyAllWindows()
            
    else:
        index = 0    
        for img in calibrator.RGB_imgs:
            rgb = cv2.imread(img)
            if calibrator.add_corners_rgb(rgb):
                print(f"RGB: Esquinas detectadas y agregadas frame = {index:05d}.")
                ir = cv2.imread(calibrator.IR_imgs[index], cv2.IMREAD_GRAYSCALE)
                if calibrator.add_corners_ir(ir):
                    print(f"IR: Esquinas detectadas y agregadas frame = {index:05d}.")
                else:
                    print(f"IR: No se detectaron esquinas frame = {index:05d}, imagen removida del array.")
                    calibrator.objpoints_rgb.pop()
                    calibrator.imgpoints_rgb.pop()
            else:
                print(f"RGB: No se detectaron esquinas frame = {index:05d}, imagen removida del array.")
            index += 1
        image_shape_rgb = (rgb.shape[1], rgb.shape[0])
        #for img in calibrator.IR_imgs:
            #ir = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            #calibrator.add_corners_ir(ir)
            #print("IR: Esquinas detectadas y agregadas.")
        image_shape_ir = (ir.shape[1], ir.shape[0])

    print("Calibrando camaras individuales...")
    ret_rgb, mtx_rgb, dist_rgb, _, _ = calibrator.calibrate_camera(calibrator.objpoints_rgb, calibrator.imgpoints_rgb, image_shape_rgb)
    ret_ir, mtx_ir, dist_ir, _, _ = calibrator.calibrate_camera(calibrator.objpoints_ir, calibrator.imgpoints_ir, image_shape_ir)
    
    print("Calibracion RGB Error:", ret_rgb)
    print("Calibracion IR Error:", ret_ir)
    
    print("objpoints RGB e IR: ", len(calibrator.objpoints_rgb), len(calibrator.objpoints_ir))
    print("img points RGB e IR: ", len(calibrator.imgpoints_rgb), len(calibrator.imgpoints_ir))

    print("Calibrando stereo...")
    ret_stereo, R, T, E, F = calibrator.calibrate_stereo(mtx_rgb, dist_rgb, mtx_ir, dist_ir, image_shape_rgb)
    print("Calibracion Stereo Error:", ret_stereo)
    print("Matriz de Rotacion:", R)
    print("Vector de Traslacion:", T)

    with open("calibration.txt", "w") as file:
        file.write("ret_rgb: " + str(ret_rgb) + "\n")
        file.write("mtx_rgb: " + str(mtx_rgb) + "\n")
        file.write("dist_rgb: " + str(dist_rgb) + "\n")
        file.write("ret_ir: " + str(ret_ir) + "\n")
        file.write("mtx_ir: " + str(mtx_ir) + "\n")
        file.write("dist_ir: " + str(dist_ir) + "\n")
        file.write("ret_stereo: " + str(ret_stereo)  + "\n")
        file.write("R: " + str(R) + "\n")
        file.write("T: " + str(T) + "\n")
        file.write("E: " + str(E) + "\n")
        file.write("F: " + str(F) + "\n")
    file.close()

    calibrator.release()
    cv2.destroyAllWindows()
