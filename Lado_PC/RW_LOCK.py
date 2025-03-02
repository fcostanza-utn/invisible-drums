import multiprocessing
"""
CLASS RWLOCK
"""
class RWLock:
    def __init__(self):
        # Contador de procesos lectores (compartido entre procesos)
        self.readers = multiprocessing.Value('i', 0)
        # Lock para proteger la modificación del contador de lectores
        self.readers_lock = multiprocessing.Lock()
        # Lock que se adquiere para escritura o cuando hay al menos un lector
        self.resource_lock = multiprocessing.Lock()

    def acquire_read(self):
        # Se adquiere el lock para modificar el contador de lectores
        with self.readers_lock:
            self.readers.value += 1
            # Si es el primer lector, bloquea el recurso para evitar escrituras
            if self.readers.value == 1:
                self.resource_lock.acquire()

    def release_read(self):
        # Se libera el lock para modificar el contador de lectores
        with self.readers_lock:
            self.readers.value -= 1
            # Si es el último lector, libera el recurso
            if self.readers.value == 0:
                self.resource_lock.release()

    def acquire_write(self):
        # El escritor adquiere directamente el recurso, garantizando exclusividad
        self.resource_lock.acquire()

    def release_write(self):
        self.resource_lock.release()