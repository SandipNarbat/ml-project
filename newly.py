import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fake_currency_model.h5')

# Create a Tkinter window
window = tk.Tk()
window.title("Fake Currency Detection")

# Function to perform detection
def detect_currency():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path).convert('L')
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 64, 64, 1)

    result = model.predict(image)
    if result[0][0] > 0.5:
        label.config(text="Fake Currency")
    else:
        label.config(text="Genuine Currency")

# Create a browse button
browse_button = tk.Button(window, text="Browse Image", command=detect_currency)
browse_button.pack()

# Create a label for the result
label = tk.Label(window, text="", font=("Helvetica", 16))
label.pack()

# Start the Tkinter main loop
window.mainloop()
