# Tutorial 2 - Blood Pressure

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def calculate_one_patient(cuff_pressure):
    
    # specify
    dt = 0.01

    # isolate the deflation phase:
    # find the highest pressure, remove anything before
    index_max = np.argmax(cuff_pressure)
    cuff_pressure = cuff_pressure[index_max:]
       
    # find local peaks (generates array of indices of peaks)
    # each element in the array indicates which index a peak is found
    peaks,_ = find_peaks(cuff_pressure)
    
    # find the differences between each peak
    # np.diff() returns the differences between each element
    differences = np.diff(peaks)
    
    # hence the differences multiplied by the sampling rate,
    # gives the time period
    # calculate the time periods (0.01s sampling rate)
    periods = dt * differences
    
    # averages the periods for one patient
    avg_period = np.mean(periods)
    
    # each element in the periods array is the calculated
    # period based on one waveform, hence each item in the
    # BPM array would be the computed BPM, based on one waveform
    # calculate the BPMs
    bpm = 1 / periods * 60
    
    # averages the BPM for one patient
    avg_bpm = np.mean(bpm)
    
    # iterate thru the waveform to perform averaging 
    # and get the oscillation wave
    # similar to high-pass filtering
    bound = round(avg_period*0.5/dt)
    filtered_cuff_pressure = np.zeros(len(cuff_pressure))
    
    for j in range(bound, len(cuff_pressure)-bound):
        # get upper and lower bounds
        upp_idx = j + bound
        low_idx = j - bound
        
        avg_pressure = np.mean(cuff_pressure[low_idx:upp_idx])
        filtered_cuff_pressure[j] = cuff_pressure[j] - avg_pressure
        
    # Obtain the oscillogram
    peaks_pos,_ = find_peaks(filtered_cuff_pressure)
    peaks_neg,_ = find_peaks(-filtered_cuff_pressure)
    
    osc_ampl = []
    osc_pres = []

    for k in range(0, min(len(peaks_pos), len(peaks_neg))):
        # compute amplitude of oscillation
        amp = np.abs(filtered_cuff_pressure[peaks_pos[k]] - \
                     filtered_cuff_pressure[peaks_neg[k]])
        
        # compute the pressure
        press = 0.5 * (cuff_pressure[peaks_pos[k]] + \
                       cuff_pressure[peaks_neg[k]])
        
        # append to list
        osc_ampl.append(amp)
        osc_pres.append(press)
        
    # Derivative method to obtain SBP and DBP (Systolic & Diastolic)
    sbp = dbp = MAP = 0
    max_der = -1e10
    min_der = 1e10
    max_osc = 0
    
    for l in range(1, len(osc_pres)):
        derivative = (osc_ampl[l] - osc_ampl[l-1]) /\
                     (osc_pres[l] - osc_pres[l-1])
        
        # track max and min, update sbp, dbp, and map
        if derivative > max_der:
            max_der = derivative
            # diastolic pressure is the cuff pressure at which the
            # derivative is the largest
            dbp = osc_pres[l]
        if derivative < min_der:
            min_der = derivative
            # systolic pressure is the cuff pressure at which the 
            # derivative is the lowest
            sbp = osc_pres[l]
        if osc_ampl[l] > max_osc:
            max_osc = osc_ampl[l]
            MAP = osc_pres[l]
       
    # Fixed ratio method to obtain SBP and DBP
    f_s = 0.45 
    f_d = 0.85

    max_delta_d = max_delta_s = 1e10
    dbp_fixed = sbp_fixed = 0
    delta_od = f_d * max_osc
    delta_os = f_s * max_osc
    
    for j in range(0, len(osc_pres)):
        # checking the left side of the curve
        # and that we capture the closest possible value
        if (np.abs(osc_ampl[j] - delta_od) < max_delta_d and \
            osc_pres[j] < MAP):
            max_delta_d = np.abs(osc_ampl[j] - delta_od)
            dbp_fixed = osc_pres[j]
            
        # checking the right side of the curve
        # and that we capture the closest possible value
        if (np.abs(osc_ampl[j] - delta_os) < max_delta_s and \
            osc_pres[j] > MAP):
            max_delta_s = np.abs(osc_ampl[j] - delta_os)
            sbp_fixed = osc_pres[j]  
               
    return {
        'bpm' : avg_bpm,
        'sbp_der' : sbp,
        'dbp_der' : dbp,
        'sbp_fixed' : sbp_fixed,
        'dbp_fixed' : dbp_fixed,
        'map' : MAP
        }
    
    # if x:
    # # plot deflation value
    #     time = np.arange(0, len(cuff_pressure) * dt, dt)
    #     plt.figure()
    #     plt.plot(time, cuff_pressure, 'b-', \
    #               time[peaks], cuff_pressure[peaks], 'r.')
        
    # # plot oscillation curve
    #     time = np.arange(0, len(cuff_pressure) * dt, dt)
    #     plt.figure()
    #     plt.plot(time, filtered_cuff_pressure, 'b-', \
    #               time[peaks_pos], filtered_cuff_pressure[peaks_pos], 'r.', \
    #               time[peaks_neg], filtered_cuff_pressure[peaks_neg], 'g.')
    
    # # if plot ossillation amplitude and pressure
    #     plt.figure()
    #     plt.plot(osc_pres, osc_ampl, 'r-')
        
    #     x = False
    
# def compare_results(patient_data, verification_data):
    
#     # initialise numbers
#     corr_hr = 0
#     corr_sbp = 0
#     corr_dbp = 0
#     corr_sbp_fixed = 0
#     corr_dbp_fixed = 0
    
#     # verify data
#     verify = {
#         'bpm' : False,
#         'sbp_fixed' : False,
#         'dbp_fixed' : False,
#         'sbp_der' : False,
#         'dbp_der' : False,
#         }
    
#     if (np.abs(patient_data['bpm'] - correct_data[i,2] < hr_tol)):
#         verify['bpm'] = True
#         corr_hr += 1
#     if (np.abs(patient_data['sbp_der'] - correct_data[i,0] < bp_tol)):
#         verify['sbp_der'] = True
#         corr_sbp += 1
#     if (np.abs(patient_data['sbp_fixed'] - correct_data[i,0] < bp_tol)):
#         verify['sbp_fixed'] = True
#         corr_sbp_fixed += 1
#     if (np.abs(patient_data['dbp_der'] - correct_data[i,1] < bp_tol)):
#         verify['dbp_der'] = True
#         corr_dbp += 1
#     if (np.abs(patient_data['dbp_fixed'] - correct_data[i,1] < bp_tol)):
#         verify['dbp_fixed'] = True
#         corr_dbp_fixed += 1
        
#     numbers = [corr_hr, corr_sbp, corr_dbp, corr_sbp_fixed, corr_dbp_fixed]
    
#     return verify, numbers
    

### actual computation

# retrieve data
cuff_data = np.genfromtxt("file_1.txt")
# correct_data = np.genfromtxt("correct_data.txt")

# specify tolerances
bp_tol = 3
hr_tol = 1.5

# storage
patients = []
verification = []
numbers = {
    'corr_hr' : 0, 
    'corr_sbp' : 0,
    'corr_dbp' : 0,
    'corr_sbp_fixed' : 0,
    'corr_dbp_fixed' : 0
    }

for i in range(0, cuff_data[0].size):  # iterate thru the patients
    # extract each patient, alls rows, specific column  
    cuff_pressure = cuff_data[:, i]
    patient_data = calculate_one_patient(cuff_pressure)
    patients.append(patient_data)
    
    
    # verification_data = correct_data[i, :]
    
    # verify, numbers = compare_results(patient_data, verification_data)
    # verification.append(verify)
        
    

    

        
      
    
    