# Author: Rachael Judy
# File: equalizerGUI.py
# Purpose: a GUI interface for an equalizer board + filtering for events
# Date: 11/26/21
# Notes:
#   slider labels go 60, 150, 400, 1k, 2.4k, 15k
#   need to determine what chunks to feed the audio into the filters with

# general libraries
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.signal
import threading

from tkinter import filedialog
from tkinter import *

# for interaction with wav files
import scipy.io.wavfile as wav
import playsound


# global filepath being filtered and sampling frequency
filepath = ''
F_s = 10250  # needs to come from wave file data as varies


# GUI methods
def choose_file():
    try:
        global filepath
        filepath = filedialog.askopenfilename(filetypes=[('Audio Files', '*.wav')])
        filename_label.config(text=filepath)
    except ValueError:
        pass


def get_slider_values():
    slider_vector = [value60hz.get(), value150hz.get(), value400hz.get(),
                     value1khz.get(), value2_4khz.get(), value15khz.get()]
    return slider_vector


# set board to correspond to the boost coefficients for analog values (cycles/s)
def set_gains(gains):
    value60hz.set(gains[0])
    value150hz.set(gains[1])
    value400hz.set(gains[2])
    value1khz.set(gains[3])
    value2_4khz.set(gains[4])
    value15khz.set(gains[5])


def play(filename=''):
    global filepath
    if filename == '':
        playsound.playsound('result.py')
    else:
        playsound.playsound(filename)


# button event handlers for defaults
def high_pushed():
    set_gains([2, 2, 2, 4, 6, 6])


def low_pushed():
    set_gains([6, 6, 5, 2, 2, 2])


def bass_pushed():
    set_gains([6.5, 6, 5.5, 4.5, 2.5, 2.5])


def vocals_pushed():
    set_gains([2, 2.5, 5.5, 6, 5.5, 4])


def volume_pushed():
    set_gains([7, 6, 5.5, 6, 5.5, 6.5])


# returns np array of shape (length, ) of type complex rectangular
def get_numpy_array_from_wav(filename, seconds=10, start=3):
    # conversion from stored wave file to output array
    # Read file to get buffer
    global F_s
    F_s, data = wav.read(filename)
    length = pow(2, math.ceil(math.log(seconds*F_s) / math.log(2)))

    data = data[start*F_s:start*F_s + length]
    x = np.zeros((data.shape[0],), dtype=np.complex_)
    for i in range(data.shape[0]):
        x[i] = data[i][0] + 1j*data[i][1]
    return x


# writes back complex array passed in
def write_numpy_to_wave(array, rate, filename='result.wav'):
    output = np.zeros((len(array), 2), dtype=np.int16)
    for i in range(len(array)):
        output[i] = [array[i].real, array[i].imag]
    wav.write(filename, rate, output)


def create_filter_from_constants_freq_sample(N=512):
    # must have sampling frequency at least twice the highest component
    global F_s  # samples/s

    # frequencies go from 0 to 2pi (0 to pi for unique)
    # w = 2*pi*F/F_s
    # create N point filter by linearizing between set points
    gains = get_slider_values()
    freq = np.zeros(N // 2)
    step = F_s / N
    # sampling frequency at omega=pi split into N steps needs omega/2pi * F_s as max possible freq
    # since going to be reflected at pi, steps must be twice the seconds
    arr_pos = 0
    for i in range(0, int(60 / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = 0
            arr_pos = arr_pos + 1
    for i in range(0, int((150 - 60) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[0] + (gains[1] - gains[0]) / (150 - 60) * step * i
            arr_pos = arr_pos + 1
    for i in range(0, int((400 - 150) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[1] + (gains[2] - gains[1]) / (400 - 150) * step * i
            arr_pos = arr_pos + 1
    for i in range(0, int((1000 - 400) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[2] + (gains[3] - gains[2]) / (1000 - 400) * step * i
            arr_pos = arr_pos + 1
    for i in range(0, int((2400 - 1000) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[3] + (gains[4] - gains[3]) / (2400 - 1000) * step * i
            arr_pos = arr_pos + 1
    for i in range(0, int((15000 - 2400) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[4] + (gains[5] - gains[4]) / (15000 - 2400) * step * i
            arr_pos = arr_pos + 1
    for i in range(0, int((F_s / 2 - 15000) / step)):
        if arr_pos < N // 2:
            freq[arr_pos] = gains[5] + (0 - gains[5]) / (F_s / 2 - 15000) * step * i
            arr_pos = arr_pos + 1

    freq = np.append(freq, freq[::-1])

    # use freq sampling method
    return np.fft.ifft(freq, N) * 100


def create_filter_from_constants_least_squared(N=2048):
    global F_s

    sliders = get_slider_values()

    bands = [0, 50, 60, 150, 400, 1000, 2400, 15000, 15500, F_s/2]
    desired = [0, 0]
    desired.extend(sliders)
    desired.extend([0, 0])

    filter = scipy.signal.firls(N+1, bands, desired, fs=F_s)
    return filter


def test_plot(y, title=''):
    plt.plot(np.linspace(0, y.shape[0], y.shape[0], False), y)
    plt.title(title)
    plt.xlabel('n')
    plt.ylabel('Magnitude')
    plt.show()


def dtft_plot(y, size=2**8, title=''):
    dtft = np.fft.fft(y, size)
    plt.plot(np.linspace(0, 2*math.pi, len(dtft), True), np.abs(dtft))
    plt.title(title + ' Magnitude')
    plt.xlabel('omega (rad/s)')
    plt.ylabel(f'|H|')
    plt.show()


# main functionality that uses others to set up
def filter_signal():
    global F_s, filepath
    # convolve in time domain and convert then play and
    if length_entry.get() != '':
        length = int(length_entry.get())
    else:
        length = 30
    x = get_numpy_array_from_wav(filepath, length)

    # test output
    write_numpy_to_wave(x, F_s, 'input samples.wav')
    test_plot(x, 'x')
    dtft_plot(x, x.shape[0], 'X')

    # h = create_filter_from_constants_freq_sample(2**10)
    h = create_filter_from_constants_least_squared(2**7) / 2

    test_plot(h,  'Filter h ')
    dtft_plot(h, h.shape[0]-1, 'Filter H ')

    y = np.convolve(x, h)

    # display code
    print('F_s: ', F_s)
    print('x length: ', x.shape)
    print('h length: ', h.shape)
    print('y length: ', y.shape)

    test_plot(y, title='y(filtered)')
    dtft_plot(y, x.shape[0], title='Result Y')

    filename = filepath[:-4] + '_result.wav'
    write_numpy_to_wave(y, F_s, filename)

    # start a thread so other stuff can keep going, play the audio file
    audio_thread = threading.Thread(target=play, args=[filename])
    audio_thread.start()


# set up graphic equalizer GUI
root = Tk()
root.title("Graphic Equalizer")
# Initiation of Frames
select_frame = Frame(root)
select_frame.pack(side=TOP)

# creation of sliders
sliderframes = Frame(root)
sliderframes.pack(side=TOP)
slider1frame = Frame(sliderframes)
slider1frame.pack(side=LEFT)
slider2frame = Frame(sliderframes)
slider2frame.pack(side=LEFT)
slider3frame = Frame(sliderframes)
slider3frame.pack(side=LEFT)
slider4frame = Frame(sliderframes)
slider4frame.pack(side=LEFT)
slider5frame = Frame(sliderframes)
slider5frame.pack(side=LEFT)
slider6frame = Frame(sliderframes)
slider6frame.pack(side=LEFT)
lengthframe = Frame(root)
lengthframe.pack(side=RIGHT)

# creation of import frame
importframe = Frame(root)
importframe.pack(side=BOTTOM)

# Value creation
value60hz = DoubleVar()
value150hz = DoubleVar()
value400hz = DoubleVar()
value1khz = DoubleVar()
value2_4khz = DoubleVar()
value15khz = DoubleVar()

# All sliderframes widgets
# Slider 1
w = Scale(slider1frame, from_=7, to=0, variable=value60hz)
w.set(0)
w.pack(side=TOP)
# Label for slider 1
label60hz = Label(slider1frame, text='60Hz')
label60hz.pack(side=TOP)

# Slider 2
w2 = Scale(slider2frame, from_=7, to=0, variable=value150hz)
w2.set(0)
w2.pack(side=TOP)
# Label for slider 2
label150hz = Label(slider2frame, text='150Hz')
label150hz.pack(side=TOP)

# Slider 3
w3 = Scale(slider3frame, from_=7, to=0, variable=value400hz)  #
w3.set(0)
w3.pack(side=TOP)
# Label for slider 3
label400hz = Label(slider3frame, text='400Hz')
label400hz.pack(side=TOP)

# Slider 4
w4 = Scale(slider4frame, from_=7, to=0, variable=value1khz)  #
w4.set(0)
w4.pack(side=TOP)
# Label for slider 4
label1khz = Label(slider4frame, text='1kHz')
label1khz.pack(side=TOP)

# Slider 5
w5 = Scale(slider5frame, from_=7, to=0, variable=value2_4khz)  #
w5.set(0)
w5.pack(side=TOP)
# Label for slider 5
label2_4khz = Label(slider5frame, text='2.4kHz')
label2_4khz.pack(side=TOP)

# Slider 6
w6 = Scale(slider6frame, from_=7, to=0, variable=value15khz)  ##
w6.set(0)
w6.pack(side=TOP)
# Label for slider 5
label15khz = Label(slider6frame, text='15kHz')
label15khz.pack(side=TOP)

# Audio Import Button
audio_button = Button(importframe, text='Import Audio...', command=choose_file)
audio_button.pack(side=LEFT)
filename_label = Label(importframe, text='No File Chosen.')
filename_label.pack(side=LEFT)
play_button = Button(importframe, text='Play', command=filter_signal)
play_button.pack(side=LEFT)

# top side buttons for choices
choice_label = Label(select_frame, text="Choose audio effect")
choice_label.pack(side=TOP)
high_button = Button(select_frame, text="Highs", command=high_pushed)
high_button.pack(side=LEFT)
lows_button = Button(select_frame, text="Lows", command=low_pushed)
lows_button.pack(side=LEFT)
bass_button = Button(select_frame, text="Bass", command=bass_pushed)
bass_button.pack(side=LEFT)
vocals_button = Button(select_frame, text="Vocals", command=vocals_pushed)
vocals_button.pack(side=LEFT)
volume_button = Button(select_frame, text="Volume", command=volume_pushed)
volume_button.pack(side=LEFT)
custom_button = Button(select_frame, text="Custom (on sliders)")
custom_button.pack(side=LEFT)

# entry box for start and seconds
length_label = Label(sliderframes, text="Seconds to filter")
length_label.pack(side=TOP)
length_entry = Entry(sliderframes)
length_entry.pack(side=TOP)

# use this to get the values
mainloop()
