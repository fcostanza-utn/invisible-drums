import mido
import pygame

# Inicializa Pygame
pygame.init()

# Carga los sonidos de batería
sounds = {
    36: pygame.mixer.Sound("./Samples/kick.wav"),   # Nota MIDI 36 para el kick drum
    38: pygame.mixer.Sound("./Samples/snare.wav"),  # Nota MIDI 38 para la caja
    42: pygame.mixer.Sound("./Samples/hihat.wav"),  # Nota MIDI 42 para el hi-hat cerrado
    49: pygame.mixer.Sound("./Samples/crash.wav"),  # Nota MIDI 49 para el crash
    51: pygame.mixer.Sound("./Samples/ride.wav"),  # Nota MIDI 51 para el ride
    50: pygame.mixer.Sound("./Samples/hightom.wav"),  # Nota MIDI 50 para el tom alto
    45: pygame.mixer.Sound("./Samples/lowtom.wav"),  # Nota MIDI 45 para el tom bajo
}

# Función para reproducir el sonido correspondiente a la nota MIDI
def play_drum(note, volume):
    if note in sounds:
        sounds[note].play()
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
                print(f"Received note: {msg.note}")
                play_drum(msg.note, (msg.velocity/100))
    except KeyboardInterrupt:
        print("Stopped by user.")
