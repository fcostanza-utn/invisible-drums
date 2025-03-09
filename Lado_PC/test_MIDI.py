import keyboard
import mido

# Abrir puerto MIDI
# Si tienes más de un puerto, puedes listar los disponibles con mido.get_output_names()
# y seleccionar el que prefieras.
portmidi = mido.Backend('mido.backends.rtmidi')
outport = portmidi.open_output('MIDI1 1')

# Mapeo de teclas a notas MIDI (puedes modificarlo según tus necesidades)
key_to_note = {
    'a': 36,  # C4
    's': 38,
    'd': 42,
    'f': 49,
    'g': 51,
    'h': 50,
    'j': 45,
}

def send_note_on(note):
    """Envía un mensaje 'note_on' (nota encendida) vía MIDI."""
    msg = mido.Message('note_on', note=note, velocity=100)
    outport.send(msg)
    print(f"Nota ON enviada: {note}")

def send_note_off(note):
    """Envía un mensaje 'note_off' (nota apagada) vía MIDI."""
    msg = mido.Message('note_off', note=note, velocity=100)
    outport.send(msg)
    print(f"Nota OFF enviada: {note}")

def handle_key_event(event):
    # Al presionar una tecla
    if event.event_type == 'down' and event.name in key_to_note:
        send_note_on(key_to_note[event.name])
    # Al soltar una tecla
    elif event.event_type == 'up' and event.name in key_to_note:
        send_note_off(key_to_note[event.name])

# Captura los eventos de teclado
keyboard.hook(handle_key_event)

print("Presiona las teclas asignadas para enviar notas MIDI. Presiona ESC para salir.")

# Espera a que se presione la tecla ESC para finalizar el programa
keyboard.wait('esc')
