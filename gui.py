from tkinter import *
import os
from matplotlib.pyplot import show

from numpy.core.numeric import ones
from filtering import *
from tkinter import messagebox

app = Tk()

def makepath():
    #print(getpath() + '/{}/{}'.format(path_name.get(), folder_name.get()))
    return getpath() + '/{}/{}'.format(path_name.get(), folder_name.get())

def patient_sheets():
    # return pre, post, triggers of the given patient
    pre, post, trigger = splitprepost(getPatientSheetsConcat(makepath()))
    #print('fetched excel files')
    return pre, post, trigger

# computes all graphs for given patient
def compute_graphs():
    pre, post, trigger = patient_sheets()

    # big if statement for conditions, could work on a better version
    """ if (highcut.get() != 0):
        if constr_time != 0:
            make_graphs(makepath(), path_name, pre, post, trigger, highcut.get(), constr_time.get()*120)
        else:
         make_graphs(makepath(), path_name, pre, post, trigger, highcut.get())           
    else:
        if constr_time != 0:
            make_graphs(makepath(), path_name, pre, post, trigger, constr_time.get()*120)
        else: """
    make_graphs(makepath(), path_name, pre, post, trigger, highcut.get(), constr_time.get()*120, show_original.get(),
                area_06.get(), area_630.get(), latency.get(), velocity.get())
    messagebox.showinfo(message='Graphs are done!')

def y():
    print(highcut.get())

app.title('Pupil Graph Analysis')
app.geometry('800x400')


# LEFT SIDE - GRAPH MAKER - COLUMNS 0 & 1

#fetch path name
path_name = StringVar()
path_name_label = Label(app, text='Path to file:', font=('bold',14), pady = 20)
path_name_label.grid(row = 0, column=0)

#path entry
path_entry = Entry(app, textvariable=path_name)
path_entry.grid(row=0, column=1)

#folder name
folder_name = StringVar()
folder_name_label = Label(app, text='Path to folder:', font=('bold',14), pady = 20)
folder_name_label.grid(row = 1, column=0)

#folder name entry
folder_entry = Entry(app, textvariable=folder_name)
folder_entry.grid(row=1, column=1)


opt_text = Label(app, text='Optional Settings', padx=20)
opt_text.grid(row=2, column=0)

#highcut selector
highcut = IntVar()
highcut_label = Label(app, text='Highcut: ', font=('bold',14), pady=20)
highcut_label.grid(row = 3, column=0)
highcut_entry = Entry(app, textvariable=highcut)
highcut_entry.grid(row=3, column=1)

# contraction time selector
constr_time = IntVar()
constr_time_label = Label(app, text=' Max. time to min[s]', font=('bold',14), pady=20)
constr_time_label.grid(row = 4, column=0)
constr_time_entry = Entry(app, textvariable=highcut)
constr_time_entry.grid(row=4, column=1)

# show original graph toggle
show_original = IntVar()
show_original_label = Checkbutton(app, text='show original graph', variable=show_original, onvalue=1, offvalue=0)
show_original_label.grid(row = 5, column=0)



compute_btn = Button(app,text='compute graphs', width=12, command=compute_graphs)
compute_btn.grid(row=6, column=1)




# RIGHT SIDE - STATISTICS
stats = Label(app, text='Statistics', font=('bold', 14), padx=150)
stats.grid(row = 0, column=2)

area_06 = IntVar()
area_06_label = Checkbutton(text='Area under curve [0s-6s]', variable=area_06, onvalue=1, offvalue=0)
area_06_label.grid(row=1, column=2)

area_630 = IntVar()
area_630_label = Checkbutton(text='Area under curve [6s-30s]', variable=area_630, onvalue=1, offvalue=0)
area_630_label.grid(row=2, column=2)

latency = IntVar()
latency_label = Checkbutton(text='Latency', variable=latency, onvalue=1, offvalue=0)
latency_label.grid(row=3, column=2)

velocity = IntVar()
velocity_label = Checkbutton(text='Velocity', variable=velocity, onvalue=1, offvalue=0)
velocity_label.grid(row=4, column=2)


#runs app
app.mainloop()

#pyinstaller --onefile --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' gui.py