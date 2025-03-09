import threading

class DataSynchronizer:
    def __init__(self):
        self.button_right = True
        self.button_left = True

        self.button_repeat_right = False
        self.button_repeat_left = False

        self.offset_time_camera = 0
        self.offset_time_imu = 0

        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0
      
        self.master = 0
        self.lock = threading.Lock()  # Bloqueo para acceso seguro a los datos

    def update_imu_time(self, new_time):
        with self.lock:
            self.offset_time_imu = new_time

    def update_camera_time(self, new_time):
        with self.lock:
            self.offset_time_camera = new_time

    def set_offsets(self, x_right, y_right, z_right, x_left, y_left, z_left):
        with self.lock:
            print("Setting offsets: ", x_right, y_right, z_right)
            if x_right == 1000 and y_right == 1000 and z_right == 1000:
                self.x_offset = x_left
                self.y_offset = y_left
                self.z_offset = z_left
            if x_left == 1000 and y_left == 1000 and z_left == 1000:
                self.x_offset = x_right
                self.y_offset = y_right
                self.z_offset = z_right

    def set_button(self, value: bool, side: str):
        with self.lock:
            if side == 'right':
                self.button_right = value
            elif side == 'left':
                self.button_left = value

    def set_button_repeat(self, value: bool, side: str):
        with self.lock:
            if side == 'right':
                self.button_repeat_right = value
            elif side == 'left':
                self.button_repeat_left = value

    def set_master(self, value: int):
        with self.lock:
            self.master = value

    def get_state(self):
        with self.lock:
            return {
                'button_right': self.button_right,
                'button_left': self.button_left,
                'button_repeat_right': self.button_repeat_right,
                'button_repeat_left': self.button_repeat_left,
                'offset_time_camera': self.offset_time_camera,
                'offset_time_imu': self.offset_time_imu,
                'x_offset': self.x_offset,
                'y_offset': self.y_offset,
                'z_offset': self.z_offset,
                'master': self.master
            }