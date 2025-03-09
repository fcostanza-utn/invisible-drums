import mido
import pygame
import time

# Inicializa Pygame
pygame.init()

# Carga los sonidos de batería
sounds = {
    36: "./Samples/kick.wav",   # Nota MIDI 36 para el kick drum
    38: "./Samples/Ensoniq-ESQ-1-Snare.wav",  # Nota MIDI 38 para la caja
    42: "./Samples/Closed-Hi-Hat-1.wav",  # Nota MIDI 42 para el hi-hat cerrado
    49: "./Samples/Crash-Cymbal-1.wav",  # Nota MIDI 49 para el crash
    51: "./Samples/Korg-NS5R-Power-Ride-Cymbal.wav",  # Nota MIDI 51 para el ride
    50: "./Samples/Hi-Tom-1.wav",  # Nota MIDI 50 para el tom alto
    45: "./Samples/Floor-Tom-1.wav",  # Nota MIDI 45 para el tom bajo
}

# Función para reproducir el sonido correspondiente a la nota MIDI
def play_drum(note, volume):
    print("vol: ", volume)
    if note in sounds:
        sound = pygame.mixer.Sound(sounds[note])
        sound.set_volume(volume)
        sound.play()
    else:
        print(f"No sound assigned for note {note}")

print("Available MIDI input ports:")
print(mido.get_input_names())

# Configura el puerto MIDI para recibir mensajes
with mido.open_input('MIDI1 0') as port:
    print("Listening for MIDI messages...")
    try:
        for msg in port:
            # Solo procesa mensajes de nota on
            if msg.type == 'note_on' and msg.velocity > 0:
                print(f"Received note: {msg.note} VELOCITY: {msg.velocity} time: {time.time()}")
                play_drum(msg.note, float(msg.velocity/100))
            elif msg.type == 'note_off':
                print(f"Received note off: {msg.note}")
    except KeyboardInterrupt:
        print("Stopped by user.")
