from tkinter import *
import os

from tkinter import messagebox
from tkinter.filedialog import askdirectory, askopenfilename

from numpy.lib.shape_base import _put_along_axis_dispatcher

app = Tk()
path = StringVar()

def patient_sheets(path, split):
    # return pre, post, triggers of the given patient
    
    pre, post, trigger = splitprepost(getPatientSheetsConcat(path), split)
    #print('fetched excel files')
    return pre, post, trigger

# computes all graphs for given patient
def compute_graphs():
   
    pre, post, trigger = patient_sheets(path.get(),split.get())
    
    
    make_graphs(path.get(), "", pre, post, trigger, hcut.get(), constr_time.get()*120, show_original.get(),
                    area_06.get(), area_630.get(), latency.get(), velocity.get(), six_val.get(), mel.get())
    messagebox.showinfo(message='Graphs are done!')

def selectfile():
    filename = askdirectory()
    #print(filename)
    path.set(filename)
    messagebox.showinfo(message='Path is '+ path.get())


app.title('Pupil Graph Analysis')
app.geometry('800x400')


# LEFT SIDE - GRAPH MAKER - COLUMNS 0 & 1

#fetch path name
""" path_name = StringVar()
path_name_label = Label(app, text='Path to file:', font=('bold',14), pady = 20)
path_name_label.grid(row = 0, column=0) """

#path entry
""" path_entry = Entry(app, textvariable=path_name)
path_entry.grid(row=0, column=1) """

#folder name
""" folder_name = StringVar()
folder_name_label = Label(app, text='Path to folder:', font=('bold',14), pady = 20)
folder_name_label.grid(row = 1, column=0) """

#folder name entry
""" folder_entry = Entry(app, textvariable=folder_name)
folder_entry.grid(row=1, column=1) """


opt_text = Label(app, text='[Optional Settings]', padx=20)
opt_text.grid(row=0, column=0)

#highcut selector
hcut = IntVar()
highcut_label = Label(app, text='Highcut: ', font=('bold',14), pady=20)
highcut_label.grid(row = 1, column=0)
highcut_entry = Entry(app, textvariable=hcut)
highcut_entry.grid(row=1, column=1)

# contraction time selector
constr_time = IntVar()
constr_time_label = Label(app, text=' Max. time to min[s]', font=('bold',14), pady=20)
constr_time_label.grid(row = 2, column=0)
constr_time_entry = Entry(app, textvariable=constr_time)
constr_time_entry.grid(row=2, column=1)

# show original graph toggle
show_original = IntVar()
show_original_label = Checkbutton(app, text='Show Original Graph', variable=show_original, onvalue=1, offvalue=0)
show_original_label.grid(row = 3, column=1)



compute_btn = Button(app,text='Select Folder', width=12, command=selectfile)
compute_btn.grid(row=5, column=1)

split = IntVar()
split_label = Checkbutton(text='Excel Files are split', variable=split, onvalue=1, offvalue=0)
split_label.grid(row=4, column=1)

mel = IntVar()
mel_label = Checkbutton(text='Melatonine test', variable=split, onvalue=1, offvalue=0)
mel_label.grid(row=7, column=1)

path_label = Label(app, text='Select the XXX folder before exports', font=('bold', 12), pady=20)
path_label.grid(row=5, column=0)

compute_btn = Button(app,text='Compute Graphs', width=12, command=compute_graphs)
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

six_val = IntVar()
six_val_label = Checkbutton(text='Value 6s after trigger', variable=six_val, onvalue=1, offvalue=0)
six_val_label.grid(row=5, column=2)



import os
import pandas as pd
import numpy as np
from numpy import fft as fft
import matplotlib.pyplot as plt
import scipy.integrate as integr

# PATH GETTER ========================================
def getpath():
    dir_path = os.path.dirname(os.path.realpath("filtering.py"))
    return dir_path


# SHEET GETTER ========================================
def getPatientSheetsConcat(path):
    all_sheets = []
    # fetch from files

    try:
        excelFile = pd.ExcelFile(path+"/exports/000/eye0 all.xlsx")
    except FileNotFoundError:
        
        messagebox.showerror(message="No such path")
        return

    sheets= list(range(1, len(excelFile.sheet_names)-1))
    # creates list of files
    all_numpys = []
    for i in sheets:
        f = pd.read_excel(excelFile, sheet_name= str(i),dtype={'pupil_timestamp': float,
                                    'diameter': float})
        n = f.to_numpy()
        
        all_sheets.append(f)
        all_numpys.append(n)
        
    last = []
    return all_numpys


# HELPER FUNCTIONS ========================================

def compute_baseline(pre, trigger_time):
    #using Astrid's formula
    return (np.mean(pre[0:30]))

#-----------------------
def fill_nan(array):
    try:
        a = np.copy(array)
        nan_idx = pd.isnull(a)
        good_idx = np.logical_not(nan_idx)
        good_data = a[good_idx]
        interpolated = np.interp(nan_idx.nonzero()[0], 
                                good_idx.nonzero()[0], 
                                np.array(good_data, dtype='float64'))
        b = a
        b[nan_idx] = interpolated
        return b
    except ValueError:
        messagebox.ERROR

#-----------------------
def connect(a, b):
    return np.concatenate((a, b))

#-----------------------
def highcut(array, hcut):
    a = np.copy(array)
    a = np.where(a > hcut, a, np.nan)
    return fill_nan(a)

#-----------------------
def min_and_value(array):
    m = np.argmin(array)
    return m, array[m]

# SHEET SPLITTER ========================================

def splitprepost(sheets, split):
    pre_triggers = []
    post_triggers = []
    triggers = []
    if split == 0:
        for i in range(len(sheets)):
            sheet = sheets[i]
            trigger_idx = np.where(sheets[0][:,2] == 'S')[0][0]
            pre_triggers.append(fill_nan(sheet[:,1][:trigger_idx]))
            post_triggers.append(fill_nan(sheet[:,1][trigger_idx:]))
            triggers.append(trigger_idx)
        print(type(pre_triggers[0][0]))
        print(type(post_triggers[0]))
        print(type(triggers[0]))
    else:
        for i in range(len(sheets)):
            print(i)
            sheet = sheets[i] 
            
            if i %2 == 0:
                post = sheets[i+1][:,1]
                # fetches pre-trigger DataFrame
                current_array = fill_nan(sheet[:,1])
                
                trigger_idx = np.where(sheets[0][:,2] == 'S')[0][0]
                #print(trigger_idx)
                
                current_array = fill_nan(sheet[:,1])
                
                #mean = np.repeat(np.mean(current_array), current_array.shape[0])
                pre_triggers.append(current_array[:trigger_idx])
                post = np.append(current_array[trigger_idx:], post)
                post_triggers.append(post)
                triggers.append(trigger_idx)
        
    return np.copy(pre_triggers), np.copy(post_triggers), np.copy(triggers)

# FOURIER FILTER ========================================

def fourier_filter(array, fcut = 40, samples_to_add = 2000):
    a = np.copy(array) # input array

    #ZERO PADDING input signal 
    samples_to_add = 2000
    a = np.concatenate((np.ones(samples_to_add)*a[0], a, np.ones(samples_to_add)*a[-1]))
    total_samples = a.shape[0]
    # "resizing" if required for frequency domain plot
    if total_samples %2 != 0:
        total_samples -=1
        a = a[:-1]

    # first-order filter we are using
    filt = np.concatenate((np.zeros(int(total_samples/2-fcut)), 
                      np.linspace(0, 1, fcut), 
                      np.linspace(1, 0, fcut),
                      (np.zeros(int(total_samples/2-fcut)))))

    # compute Fourier Transform and show frequency domain, shifts for correct plot
    ft = np.fft.fft(a)
    ftshifted = np.fft.fftshift(ft)


    # make sure, should always pass though
    assert(ftshifted.shape[0] == filt.shape[0])

    result = (ftshifted*filt)
    return fft.ifft(fft.fftshift(result))[samples_to_add:-samples_to_add].real



# GRAPH MAKER =========================================================
def plot_graph():
    a = np.ones(100)
    plt.plot(a, '--', label='baseline', color='gray')


def make_graphs(save_path, patient_name, pre, post, trigger, hcut, constr_time, show_original,
                area_06, area_630, latency, velocity, six_val, mel):

    # setting default arguments
    if hcut == 0:
        hcut = 40
    if constr_time == 0:
        constr_time = 400

    #make sure it's legit
    
    
    assert(len(pre) == len(trigger))
    assert(len(post) ==len(pre))
    


    all_data =""

    for i in range(len(post)):
        post_ = highcut(np.copy(post[i]), hcut)
        pre_ = np.copy(pre[i])
        current_min = post_[0]


        
        #process post_ with simple derivative filter 
        for j in range(1, constr_time):
            if post_[j] <= current_min and post_[j] != np.nan:
                current_min = post_[j]
                
            if post_[j] > current_min:
                post_[j] = np.nan
                p = fill_nan(post_) # linear interpolation
                
        ft = fourier_filter(p, fcut = 150)
        plt.figure(figsize=(10,5))
        
        baseline = np.ones(
            pre[i].shape[0]+post[i].shape[0])*compute_baseline(fill_nan(pre[i]), trigger[i])    


        # PLOTTING 

        plt.plot(baseline, '--', label='baseline', color='gray')
        
        if(show_original == 1):
            plt.plot(connect(pre_, np.copy(post[i])), label='original', color='green') #original graph
            plt.plot(connect(pre_, np.copy(p)), label='filtered', color='blue') #original graph

        #plt.plot(connect(pre_, post_), label='filtered', color='g')
        plt.plot(connect(pre_, ft), label='processed', color='r') # processed graph
        
        
        
        plt.plot(trigger[i], baseline[trigger[i]], 'x',mew=2, ms=5, label='trigger', color='black')
        
        
        mindex, min_value = min_and_value(ft[:constr_time])
        plt.plot(mindex, min_value, '+', label= 'min value', color='black', mew=5,ms=10)
        plt.title(label=patient_name)
        
        plt.legend()

        #saving image & making Data File
        frames_to_min = (mindex-trigger[i])
        time_to_min = (1/120)*(mindex-trigger[i])        
        
        save_path_graphs = save_path+'/results/processed_'+str(i)

        if not os.path.exists(save_path+'/results'):
            os.mkdir(save_path+'/results')
        plt.savefig(fname=save_path_graphs)

        abs_min = np.min(ft[:400])
        max_constr_ampl = baseline[0] - abs_min


        # DATA PRINTING
        all_data +="TRIGGER NUMBER [{}] : \n".format(i+1) 
        all_data += "   - Time to min from trigger: ~{:.2f} sec \n".format(time_to_min)


        all_data += '   - Maximal Constriction Amplitude [%] : {:.4f} \n'.format(max_constr_ampl)
        if latency == 1:
                all_data += '   - Latency : '
                all_data += '{:.4f} \n'.format(abs_min)
                #write_data(save_path, 'Latency', str(abs_min))
        if velocity == 1:
            all_data += '   - Velocity : '
            all_data += '{:.4f} \n'.format(max_constr_ampl/abs_min)

        if six_val == 1:
            all_data += '   - Value at 6 seconds : '
            all_data += '{:.4f} \n'.format(ft[6*120])

        print(ft.shape)
        x_axis = np.ones(ft.shape[0])*baseline[0]

        if area_06 == 1:
            all_data += '   - Area under graph [0,6]s : '
            area = compute_area(ft[:6*120])
            rect = compute_area(baseline[:6*120])
            all_data += '{:.4f} \n'.format(rect - area)

        if area_630 == 1:
            all_data += '   - Area under graph [6,30]s : '
            area = compute_area(ft[6*120:30*120])
            rect = compute_area(baseline[6*120:30*120])
            all_data += '{:.4f} \n'.format(rect - area)

        all_data += '\n'

    #write data 
    write_data(save_path, all_data)
def compute_area(y):
    return integr.trapz(y)


def write_data(path,data):
    with open(path+"/data.txt", "w") as f:
            f.write('\n')
            f.write(data)
            f.close()
    

#runs app
app.mainloop()

#pyinstaller --onefile --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' gui.py
#python setup.py py2app