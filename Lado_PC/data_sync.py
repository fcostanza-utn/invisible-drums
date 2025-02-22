import threading

class DataSynchronizer:
    def __init__(self):
        self.button = True
        self.offset_time_camera = 0
        self.offset_time_imu = 0
        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0
        self.lock = threading.Lock()  # Bloqueo para acceso seguro a los datos

    def update_imu_time(self, new_time):
        with self.lock:
            self.offset_time_imu = new_time

    def update_camera_time(self, new_time):
        with self.lock:
            self.offset_time_camera = new_time

    def set_offsets(self, x, y, z):
        with self.lock:
            self.x_offset = x
            self.y_offset = y
            self.z_offset = z

    def set_button(self, value: bool):
        with self.lock:
            self.button = value

    def get_state(self):
        with self.lock:
            return {
                'button': self.button,
                'offset_time_camera': self.offset_time_camera,
                'offset_time_imu': self.offset_time_imu,
                'x_offset': self.x_offset,
                'y_offset': self.y_offset,
                'z_offset': self.z_offset
            }