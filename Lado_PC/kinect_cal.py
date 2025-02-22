import cv2
import numpy as np
import os
from openni import openni2  # Requires the OpenNI Python wrapper

class KinectCalibrator:
    def __init__(self, openni_path="C:/Program Files/OpenNI2/Redist",chessboard_size=(9, 6), square_size=0.025):
        """
        chessboard_size: number of inner corners per chessboard row and column.
        square_size: size of a square in your defined unit (meters, for example).
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... scaled by square_size
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Lists to store object points and image points from all images.
        self.objpoints = []         # 3d point in real world space
        self.imgpoints_rgb = []     # 2d points in RGB image plane.
        self.imgpoints_ir = []      # 2d points in IR image plane.

        # Directory to save the images
        self.RGB_dir = "./Fotos_cal_RGB"
        if not os.path.exists(self.RGB_dir):
            os.makedirs(self.RGB_dir)
        self.IR_dir = "./Fotos_cal_IR"
        if not os.path.exists(self.IR_dir):  
            os.makedirs(self.IR_dir)

        # Initialize OpenNI and Kinect streams
        openni2.initialize(openni_path)
        modeRGB = openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,  #formato de 24 bits
            resolutionX=640,
            resolutionY=480,
            fps=60
        )
        modeIR = openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_GRAY16,  #formato de 16 bits
            resolutionX=640,
            resolutionY=480,
            fps=60
        )
        self.device = openni2.Device.open_any()
        self.rgb_stream = self.device.create_color_stream()
        self.ir_stream = self.device.create_ir_stream()
        self.rgb_stream.set_video_mode(modeRGB)
        self.ir_stream.set_video_mode(modeIR)
        self.rgb_stream.start()
        self.ir_stream.start()

        color_mode = self.rgb_stream.get_video_mode()
        ir_mode = self.ir_stream.get_video_mode()
        print("Color mode:", color_mode)
        print("IR mode:", ir_mode)

        self.frame_count = 0

    def capture_frame(self):
        """
        Captures a frame from both the RGB and IR streams.
        Returns:
            rgb_image: color image (BGR format)
            ir_image: IR image (grayscale, may be 16-bit depending on device)
        """
        rgb_frame = self.rgb_stream.read_frame()
        ir_frame = self.ir_stream.read_frame()

        # Create numpy arrays from the buffers.
        rgb_data = np.array(rgb_frame.get_buffer_as_triplet()).reshape(rgb_frame.height, rgb_frame.width, 3)
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        # Convert the IR frame into a grayscale image (if necessary, adjust conversion based on your device)
        ir_data = np.array(ir_frame.get_buffer_as_uint16()).reshape(ir_frame.height, ir_frame.width)
        # Normalize IR for display if needed.
        ir_image = cv2.convertScaleAbs(ir_data, alpha=(255.0/np.max(ir_data)))
        
        # rgb_data = cv2.medianBlur(rgb_data, 5)
        # ir_image = cv2.medianBlur(ir_image, 5)
        # rgb_data = cv2.bilateralFilter(rgb_data, 9, 75, 75)
        # ir_image = cv2.bilateralFilter(ir_image, 9, 75, 75)
        rgb_data = cv2.GaussianBlur(rgb_data, (5, 5), 0)
        ir_image = cv2.GaussianBlur(ir_image, (5, 5), 0)

        return rgb_data, ir_image

    def add_corners(self, rgb_image, ir_image, show_result=False):
        """
        Detects chessboard corners in both images.
        If found in both, appends the detected corners and world coordinates for calibration.
        """
        # Convert RGB to grayscale
        gray_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # IR image assumed to be grayscale
        gray_ir = ir_image

        found_rgb, corners_rgb = cv2.findChessboardCorners(gray_rgb, self.chessboard_size, None)
        found_ir, corners_ir = cv2.findChessboardCorners(gray_ir, self.chessboard_size, None)

        print("Found RGB:", found_rgb, "Found IR:", found_ir)

        if found_rgb and found_ir:
            # Refine corners for accurate calibration.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_rgb = cv2.cornerSubPix(gray_rgb, corners_rgb, (11, 11), (-1, -1), criteria)
            corners_ir = cv2.cornerSubPix(gray_ir, corners_ir, (11, 11), (-1, -1), criteria)

            if show_result:
                cv2.drawChessboardCorners(rgb_image.copy(), self.chessboard_size, corners_rgb, found_rgb)
                cv2.drawChessboardCorners(ir_image.copy(), self.chessboard_size, corners_ir, found_ir)
                cv2.imshow('RGB Calibration', rgb_image)
                cv2.imshow('IR Calibration', ir_image)
                cv2.waitKey(500)

            self.objpoints.append(self.objp)
            self.imgpoints_rgb.append(corners_rgb)
            self.imgpoints_ir.append(corners_ir)
            return True
        return False

    def calibrate_rgb(self, image_shape):
        """
        Calibrates the RGB camera.
        image_shape: (width, height) of the RGB images.
        Returns:
            ret: RMS re-projection error.
            mtx: Camera matrix.
            dist: Distortion coefficients.
            rvecs: Rotation vectors.
            tvecs: Translation vectors.
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_rgb, image_shape, None, None
        )
        return ret, mtx, dist, rvecs, tvecs

    def calibrate_ir(self, image_shape):
        """
        Calibrates the IR camera.
        image_shape: (width, height) of the IR images.
        Returns:
            ret: RMS re-projection error.
            mtx: Camera matrix.
            dist: Distortion coefficients.
            rvecs: Rotation vectors.
            tvecs: Translation vectors.
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_ir, image_shape, None, None
        )
        return ret, mtx, dist, rvecs, tvecs

    def release(self):
        """
        Stops the streams and finalizes OpenNI.
        """
        self.rgb_stream.stop()
        self.ir_stream.stop()
        openni2.unload()


if __name__ == "__main__":
    calibrator = KinectCalibrator(chessboard_size=(9, 6), square_size=0.025)
    print("Press 'c' to capture a calibration frame, 'q' to exit.")

    while True:
        rgb, ir = calibrator.capture_frame()
        cv2.imshow("RGB Live", rgb)
        cv2.imshow("IR Live", ir)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            filenameRGB = os.path.join(calibrator.RGB_dir, f"frame_{calibrator.frame_count:05d}.jpg")
            cv2.imwrite(filenameRGB, rgb)
            filenameIR = os.path.join(calibrator.IR_dir, f"frame_{calibrator.frame_count:05d}.jpg")
            cv2.imwrite(filenameIR, ir)
            calibrator.frame_count += 1
            if calibrator.add_corners(rgb, ir, show_result=True):
                print("Calibration corners detected and added.")
            else:
                print("Failed to detect calibration pattern in one or both images.")
        elif key == ord("q"):
            break

    # Assuming dimensions from the last captured images.
    image_shape_rgb = (rgb.shape[1], rgb.shape[0])
    image_shape_ir = (ir.shape[1], ir.shape[0])

    ret_rgb, mtx_rgb, dist_rgb, _, _ = calibrator.calibrate_rgb(image_shape_rgb)
    ret_ir, mtx_ir, dist_ir, _, _ = calibrator.calibrate_ir(image_shape_ir)

    print("RGB Calibration RMS error:", ret_rgb)
    print("IR  Calibration RMS error:", ret_ir)

    calibrator.release()
    cv2.destroyAllWindows()