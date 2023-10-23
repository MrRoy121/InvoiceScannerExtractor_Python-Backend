# Import the required libraries
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2

# Read input form image
img = cv2.imread('data/in.jpg')

new_size = (750, 1061)
img = cv2.resize(img, new_size)
# Find shape of the image in opencv
height, width, channel = img.shape
frame_width = width // 1
frame_height = height // 1

# Resize image opencv
img = cv2.resize(img, (frame_width, frame_height))
# Save image opencv
cv2.imwrite('resized_image.jpg', img)

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window as per input form size
win.geometry(str(frame_width) + "x" + str(frame_height))


# Define a function to draw mouse click point with tkinter
def draw_point(event):
    x1 = event.x
    y1 = event.y
    x2 = event.x
    y2 = event.y
    # Draw an oval in the given co-ordinates
    canvas.create_oval(x1, y1, x2, y2, fill="green", width=7)
    # Print mouse clicked coordinate
    print("Position = ({0},{1})".format(event.x, event.y))


# Create a canvas widget
canvas = Canvas(win, width=frame_width, height=frame_height, background="white")
canvas.pack()
canvas.place(anchor='center', relx=0.5, rely=0.5)

# Create an object of tkinter ImageTk
# Display image in tkinter canvas
img = ImageTk.PhotoImage(Image.open('resized_image.jpg'))
canvas.create_image(0, 0, image=img, anchor=NW)

canvas.grid(row=0, column=0)
canvas.bind('<Button-1>', draw_point)
# Run loop forever ImageTk
win.mainloop()