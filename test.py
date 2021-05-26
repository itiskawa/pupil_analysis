from tkinter import *
from tkinter.filedialog import askopenfilename

app = Tk()


def askfile():
    filename = askopenfilename()
    label = Label(app, text=askopenfilename)


go_btn = Button(app,text='Ôpen File', command=askfile)
go_btn.pack()


app.mainloop()