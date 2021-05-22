import os
import pandas as pd
import numpy as np
from numpy import fft as fft
import matplotlib.pyplot as plt

# PATH GETTER ========================================
def getpath():
    dir_path = os.path.dirname(os.path.realpath("filtering.py"))
    return dir_path


# SHEET GETTER ========================================
def getPatientSheetsConcat(path):
    all_sheets = []
    # fetch from files
    excelFile = pd.ExcelFile(path+"/exports/000/eye0 all.xlsx")
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

def splitprepost(sheets):
    pre_triggers = []
    post_triggers = []
    triggers = []
    for i in range(len(sheets)):
        sheet = sheets[i] 
        
        if i %2 == 0:
            post = sheets[i+1][:,1]
            # fetches pre-trigger DataFrame
            current_array = fill_nan(sheet[:,1])
            #mean = np.repeat(np.mean(current_array), current_array.shape[0])
            #pre_triggers.append(current_array)
            
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
    #fcut = 100
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
                area_06, area_630, latency, velocity):

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
        
        
        
        plt.plot(trigger[i], connect(pre_, ft)[trigger[i]], 'x',mew=2, ms=5, label='trigger', color='black')
        
        
        mindex, min_value = min_and_value(ft[:constr_time])
        plt.plot(mindex, min_value, '+', label= 'min value', color='black', mew=5,ms=10)
        plt.title(label=patient_name)
        
        plt.legend()

        #saving image & making Data File
        frames_to_min = (mindex-trigger[i])
        time_to_min = (1/120)*(mindex-trigger[i])
        
        #print("Time to min from trigger: ~{:.2f} sec - aka frame number {}".format(time_to_min, frames_to_min))
        
        
        save_path_graphs = save_path+'/results/processed_'+str(i)

        if not os.path.exists(save_path+'/results'):
            os.mkdir(save_path+'/results')
        plt.savefig(fname=save_path_graphs)

        abs_min = np.min(ft[:400])
        max_constr_ampl = baseline[0] - abs_min


        # DATA PRINTING
        all_data +="TRIGGER NUMBER [{}] : \n".format(i+1) 
        all_data += "   - Time to min from trigger: ~{:.2f} sec \n".format(i+1, time_to_min)


        all_data += '   - Maximal Constriction Amplitude [%] : {:.4f} \n'.format(max_constr_ampl)
        if latency == 1:
                all_data += '   - Latency : '
                all_data += '{:.4f} \n'.format(abs_min)
                #write_data(save_path, 'Latency', str(abs_min))
        if velocity == 1:
            all_data += '   - Velocity : '
            all_data += '{:.4f} \n'.format(max_constr_ampl/abs_min)
        all_data += '\n'
    #write data 
    """ with open(save_path+"/data.txt", "w") as f:
            f.write(all_data)
            f.close() """
    
    
    
   # write_data(save_path, 'Maximal Constriction Amplitude %', str(max_constr_ampl))
    #write_data(save_path, 'Time from Trigger to Maximal Contraction', all_data)
    
        #write_data(save_path, 'Velocity', str(max_constr_ampl/abs_min))
    write_data(save_path, all_data)

def write_data(path,data):
    with open(path+"/data.txt", "w") as f:
            f.write('\n')
            f.write(data)
            f.close()
    