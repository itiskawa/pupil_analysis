from tkinter import *
import os

from tkinter import messagebox
from tkinter.filedialog import askdirectory, askopenfilename

from numpy.lib.shape_base import _put_along_axis_dispatcher

app = Tk()
path = StringVar()

def patient_sheets(path, split, eye):
    # return pre, post, triggers of the given patient
    
    pre, post, trigger = splitprepost(getPatientSheetsConcat(path, eye), split)
    #print('fetched excel files')
    return pre, post, trigger

# computes all graphs for given patient
def compute_graphs():
   
    if (((area_06.get() == 1) or (area_630.get() == 1) or (six_val.get()==1)) and split.get() == 0):
        print("uh \n")
        messagebox.showinfo(message='Incompatible computation : code 1')
        return

    pre, post, trigger = patient_sheets(path.get(),split.get(), 0)
    
    make_graphs(path.get(), "", pre, post, trigger, hcut.get(), constr_time.get()*120, show_original.get(),
                    area_06.get(), area_630.get(), latency.get(), velocity.get(), max_constr.get(), 
                    max_vel.get(), six_val.get(), derivative.get(),  eye=0)
    messagebox.showinfo(message='Eye 0 graphs are done!')

    pre, post, trigger = patient_sheets(path.get(),split.get(), 1)
    
    
    make_graphs(path.get(), "", pre, post, trigger, hcut.get(), constr_time.get()*120, show_original.get(),
                    area_06.get(), area_630.get(), latency.get(), velocity.get(), max_constr.get(),
                    max_vel.get(), six_val.get(),derivative.get(), eye=1)
    messagebox.showinfo(message='Eye 1 graphs are done!')

def selectfile():
    filename = askdirectory()
    #print(filename)
    path.set(filename)
    messagebox.showinfo(message='Path is '+ path.get())


app.title('Pupil Graph Analysis')
app.geometry('800x400')


# LEFT SIDE - GRAPH MAKER - COLUMNS 0 & 1


# OPTIONAL SETTINGS FRAME [0,0]

opt_frame = LabelFrame(app, text='Optional Settings', padx=20, pady=20)
opt_frame.grid(row=0, column=0, padx=20, pady=20)

#highcut selector
hcut = IntVar()
highcut_label = Label(opt_frame, text='Highcut: ', font=('bold',14), pady=20)
highcut_label.grid(row = 0, column=0)
highcut_entry = Entry(opt_frame, textvariable=hcut)
highcut_entry.grid(row=0, column=1)

# contraction time selector
constr_time = IntVar()
constr_time_label = Label(opt_frame, text=' Max. time to min[s]', font=('bold',14), pady=20)
constr_time_label.grid(row = 1, column=0)
constr_time_entry = Entry(opt_frame, textvariable=constr_time)
constr_time_entry.grid(row=1, column=1)

# show original graph toggle
show_original = IntVar()
show_original_label = Checkbutton(opt_frame, text='Show Original Graph', variable=show_original, onvalue=1, offvalue=0)
show_original_label.grid(row = 2, column=0)

derivative = IntVar()
derivative_label = Checkbutton(opt_frame, text='Compute Derivative Graph', variable=derivative, onvalue=1, offvalue=0)
derivative_label.grid(row = 2, column=1)

#________________________________________________


# UNDER OPTIONAL SETTINGS

final_frame = LabelFrame(app, text='Ready', padx=20, pady=20)
final_frame.grid(row=1, column=0)

select_btn = Button(final_frame,text='Select Folder', width=12, command=selectfile)
select_btn.grid(row=0, column=0)

split = IntVar()
split_label = Checkbutton(final_frame, text='Melanopsin Test', variable=split, onvalue=1, offvalue=0)
split_label.grid(row=1, column=0)


#normal_label = Checkbutton(final_frame,text='Cone/Rod Test', variable=split, onvalue=0, offvalue=1)
#ormal_label.grid(row=1, column=1)

compute_btn = Button(final_frame,text='Compute Graphs', width=12, command=compute_graphs)
compute_btn.grid(row=2, column=0)

""" mel = IntVar()
mel_label = Checkbutton(text='Melatonine test', variable=mel, onvalue=1, offvalue=0)
mel_label.grid(row=7, column=1) """




#________________________________________________

# RIGHT SIDE - STATISTICS
stats_frame = LabelFrame(app, text='Statistics', font=('bold', 14), padx=20, pady=45)
stats_frame.grid(row = 0, column=1)

latency = IntVar()
latency_label = Checkbutton(stats_frame, text='Latency', variable=latency, onvalue=1, offvalue=0)
latency_label.grid(row=0, column=0)

velocity = IntVar()
velocity_label = Checkbutton(stats_frame, text='Velocity', variable=velocity, onvalue=1, offvalue=0)
velocity_label.grid(row=1, column=0)

max_constr = IntVar()
max_constr_label = Checkbutton(stats_frame, text='Maximal Constriction Amplitude', variable=max_constr, onvalue=1, offvalue=0)
max_constr_label.grid(row=2, column=0)

max_vel = IntVar()
max_vel_label = Checkbutton(stats_frame, text='Maximal Velocity Amplitude', variable=max_vel, onvalue=1, offvalue=0)
max_vel_label.grid(row=3, column=0)

mel_frame = LabelFrame(app, text='Melanopsin Stats', padx=20, pady=20)
mel_frame.grid(row=1, column=1)

area_06 = IntVar()
area_06_label = Checkbutton(mel_frame, text='Area under curve [0s-6s]', variable=area_06, onvalue=1, offvalue=0)
area_06_label.grid(row=0, column=0)

area_630 = IntVar()
area_630_label = Checkbutton(mel_frame, text='Area under curve [6s-30s]', variable=area_630, onvalue=1, offvalue=0)
area_630_label.grid(row=1, column=0)

six_val = IntVar()
six_val_label = Checkbutton(mel_frame, text='Value 6s after trigger', variable=six_val, onvalue=1, offvalue=0)
six_val_label.grid(row=2, column=0)



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
def getPatientSheetsConcat(path, eye):
    all_sheets = []
    # fetch from files
    ext = ''
    if eye == 0:
        ext = "/exports/000/eye0 all.xlsx"
    else:
        ext = "/exports/000/eye1 all.xlsx"
    try:
        excelFile = pd.ExcelFile(path+ext)
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

#-----------------------
def normalize_array(array, baseline):
    return array/baseline

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
        """ print(type(pre_triggers[0][0]))
        print(type(post_triggers[0]))
        print(type(triggers[0])) """
    else:
        for i in range(len(sheets)):
            #
            # print(i)
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


def make_graphs(save_path, patient_name, pre, post, trigger, hcut, constr_time_given, show_original,
                area_06, area_630, latency, velocity, max_constr, max_vel, six_val, derivative, eye):

    

    #make sure it's legit
    
    
    assert(len(pre) == len(trigger))
    assert(len(post) ==len(pre))

    data = {'Latency': [0], 'Velocity': [0], 'Max. Constriction': [0], 'Max. Velocity': [0]}
    if area_06 == 1:
        data.update({'Area 0-6': [0]})
    if area_630 == 1:
        data.update({'Area 6-30': [0]})
    if six_val == 1:
        data.update({'Value at 6s': [0]})
    #print(data)
    


    all_data =""

    for i in range(len(post)):

        post_ = highcut(np.copy(post[i]), hcut)
        pre_ = np.copy(pre[i])
        current_min = post_[0]

        # setting default arguments
        if hcut == 0:
            hcut = 40
        
        #print(len(post_))
        if len(post_) < constr_time_given:
            constr_time = len(post_)-1
        else:
            constr_time = 400
        #print(constr_time)

        
        #process post_ with simple derivative filter 
        for j in range(1, min(constr_time, len(post_))):
            if post_[j] <= current_min and post_[j] != np.nan:
                current_min = post_[j]
                
            if post_[j] > current_min:
                post_[j] = np.nan
                p = fill_nan(post_) # linear interpolation
                
        ft = fourier_filter(p, fcut = 150)

        plt.figure(figsize=(10,5))
        
        baseline = np.ones(
            pre[i].shape[0]+post[i].shape[0])*compute_baseline(fill_nan(pre[i]), trigger[i])    

        baseline_value = baseline[0]

        # PLOTTING 

        plt.plot(normalize_array(baseline, baseline_value), '--', label='baseline', color='gray')
        
        if(show_original == 1):
            plt.plot(normalize_array(connect(pre_, np.copy(post[i])), baseline_value), label='original', color='green') #original graph
            #plt.plot(connect(pre_, np.copy(p)), label='filtered', color='blue') #original graph

        processed = normalize_array(connect(pre_, ft), baseline_value)
        plt.plot(processed, label='processed', color='r') # processed graph
        
        
        
        plt.plot(trigger[i], 1, 'x',mew=2, ms=5, label='trigger', color='black')
        
        
        mindex, min_value = min_and_value(processed[:constr_time])
        plt.plot(mindex, min_value, '+', label= 'min value', color='black', mew=5,ms=10)
        
        
        plt.legend()
        eyestr = 'eye_'+str(eye)

        #saving image & making Data File
        save_path_graphs = save_path+'/results_'+eyestr+'/proc'+str(i)

        if not os.path.exists(save_path+'/results_'+eyestr):
            os.mkdir(save_path+'/results_'+eyestr)
        plt.savefig(fname=save_path_graphs)

        plt.close()

        # DERIVATIVE
        
        if derivative == 1:
            make_diff_graph(processed, save_path, eyestr, i)



        #DATA OF GRAPHS
        time_to_min = (1/120)*(mindex-trigger[i])        
        abs_min = np.min(ft[:400])
        max_constr_ampl = baseline[0] - abs_min 
        max_velocity_ampl = max_constr_ampl/time_to_min
        
        
        # DATA PRINTING
        all_data +="TRIGGER NUMBER [{}] : \n".format(i+1) 
        
        if latency == 1:
            all_data += "   - Latency to Peak: ~{:.2f} sec \n".format(time_to_min)
            data['Latency'].append(time_to_min)
            #print(data)

        if velocity == 1:
            all_data += '   - Velocity : '
            all_data += '{:.4f} \n'.format(max_constr_ampl/abs_min)
            data['Velocity'].append(max_constr_ampl/abs_min)

        if max_constr ==1:
            all_data += '   - Maximal Constriction Amplitude [%] : {:.4f} \n'.format(max_constr_ampl)
            data['Max. Constriction'].append(max_constr_ampl)

        if max_vel == 1:
            all_data += '   - Maximal Velocity Amplitude [%/s] : {:.4f} \n'.format(max_velocity_ampl)
            data['Max. Velocity'].append(max_velocity_ampl)

        if six_val == 1:
            all_data += '   - Value at 6 seconds : '
            all_data += '{:.4f} \n'.format(ft[6*120])
            data['Value at 6s'].append(ft[6*120])

        if area_06 == 1:
            all_data += '   - Area under graph [0,6]s : '
            area = compute_area(ft[:6*120])
            rect = compute_area(baseline[:6*120])
            all_data += '{:.4f} \n'.format(rect - area)
            data['Area 0-6'].append(rect-area)

        if area_630 == 1:
            all_data += '   - Area under graph [6,30]s : '
            area = compute_area(ft[6*120:30*120])
            rect = compute_area(baseline[6*120:30*120])
            all_data += '{:.4f} \n'.format(rect - area)
            data['Area 6-30'].append(rect-area)
            

        #print(data['Latency'])
        all_data += '\n'

        df = pd.DataFrame.from_dict(data)
        make_excel(save_path,eyestr, df)

    #write data 
    write_data(save_path, all_data)

def make_diff_graph(graph, path, eyestr, i):

    save_path_diff = path+'/derivatives_'+eyestr+'/proc_diff'+str(i)
    plt.figure(figsize=(10,5))


    x = np.linspace(0, graph.shape[0], graph.shape[0])
    dx = x[1]-x[0]
    y = np.copy(graph)
    dydx = np.gradient(y, dx)
    filtered_diff = fourier_filter(dydx, fcut=200)
    zero_line = np.zeros(filtered_diff.shape[0])
    #plt.plot(dydx, color='blue')
    plt.plot(zero_line, '-',color='black')
    plt.plot(filtered_diff, color='r')


    if not os.path.exists(path+'/derivatives_'+eyestr):
        os.mkdir(path+'/derivatives_'+eyestr)
    plt.savefig(fname=save_path_diff)
    plt.close()
    
def make_excel(path,eye, data):
    sheet = 'sheet'
    writer = pd.ExcelWriter(path+'/data '+eye+'.xlsx')
    data.to_excel(writer, sheet_name = sheet)
    writer.save()
    writer.close()
    #print('excel Made')

    

def compute_area(y):
    return integr.trapz(y)

def write_data(path,data_txt):

    with open(path+"/data.txt", "w") as f:
            f.write('\n')
            f.write(data_txt)
            f.close()
    

#runs app
app.mainloop()

#pyinstaller --onefile --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' gui.py
#python setup.py py2app