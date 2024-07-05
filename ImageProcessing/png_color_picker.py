from tkinter import *
from tkinter import colorchooser ,filedialog
from PIL import Image, ImageTk
import PIL

root=Tk()
root.title(" Click anywhere on image to pick a color  ")
c=Canvas(root)


def from_rgb(rgb):
    return "#%02x%02x%02x" % rgb

def colorpic(e):
    global width, height
    global rgb_label, color_display
    b1=Image.open(glb_img_name).resize((width,height)).convert("RGB")
    pixs=b1.getpixel((e.x,e.y))
    color=from_rgb((pixs))

    
    rgb_label.config(text=f"RGB: {pixs}")
    color_display.config(bg=color)
    newFrame.grid(row=0, column=1)
    
    
def width_height(owidth,oheight):
    while oheight>680:
        owidth,oheight=int(owidth/1.1),int(oheight/1.1)
    while owidth>930:
        owidth,oheight=int(owidth/1.1),int(oheight/1.1)
    return owidth,oheight


#imgname=filedialog.askopenfilename(initialdir="/Desktop/selected_images",title="open image",filetypes=(("pdf files","*.pdf")))
 # returns the filename as a string
imagename = '/Users/skylargu/Desktop/output.png'
glb_img_name=imagename
copy=Image.open(imagename)
owidth,oheight=copy.size
width,height=width_height(owidth,oheight)
myimg=ImageTk.PhotoImage(Image.open(glb_img_name).resize((width,height)))
c.config(width=width,height=height)
c.create_image(0,0,anchor=NW,image=myimg)
c.grid(row=0,column=0)
c.bind("<Button-1>",colorpic)

newFrame=Frame(root)
entry= Entry(newFrame, width= 10)
entry.grid(row=0,column=1, padx=10, pady=10)

rgb_label = Label(newFrame, text="RGB: (-, -, -)", width=20)
rgb_label.grid(row=0, column=0, padx=10, pady=5)

color_display = Label(newFrame, text="", width=10, height=2)
color_display.grid(row=1, column=0, padx=10, pady=5)

root.mainloop()
    


