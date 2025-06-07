import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import streamlit as st

# total pixel number (fixed)
aper_size = 400

#def pixel into metre
experiment_scale_mm = 1e-3
image_scale_pixel = aper_size
metre_per_pixel = experiment_scale_mm / image_scale_pixel
dx = metre_per_pixel
# dx = 2.5 micron per pixel

#set initials variables
#color = 632  # light color in nm 
color = st.slider("Wavelength (nm)", 400, 700, 633)
lamda = color*1e-9

#l = 1.25
l = st.slider("Screen Distance (m)", min_value=0.5, min_value=2.0, value=1.25, step=0.05)

#w = 20
w = int(st.slider("Slit Width (micron)", 10, 200, 50, step=2.5)/2.5)
#h = 100
h = int(st.slider("Slit Height (micron)", 10, 200, 250, step=2.5)/2.5)

#zlim = 0.1 # color adjust
zlim = st.slider("Colorscale", 0.02, 1, 0.1, step=0.02)

#create rectagular aperture
aperture = np.zeros((aper_size,aper_size))
center = (aper_size//2,aper_size//2)
aperture[center[0]-h//2:center[0]+h//2,center[1]-w//2:center[1]+w//2] = 1

diff = np.fft.fftshift(np.fft.fft2(aperture))
inten = np.abs(diff)**2
phase = np.angle(diff)

fx = np.fft.fftfreq(aper_size, d=dx)
fx = np.fft.fftshift(fx)
x_m = fx * lamda * l  # position on screen of each pixel

#convert pixel into micron
w_m = w * metre_per_pixel 
h_m = h * metre_per_pixel
#print("Aperture width in micron:", w_m*1e6)
#print("Aperture height in micron:", h_m*1e6)

diff_spatial = 1 / w_m
diff_x_m = lamda * diff_spatial * l
#print("Distance between diffraction peaks on screen (cm):", diff_x_m*100)
st.markdown("Distance between diffraction peaks on screen (cm):", diff_x_m*100)

# Plot
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.xticks(ticks=np.arange(0,aper_size,int(aper_size/5)),
           labels=np.arange(0,aper_size*metre_per_pixel*1e6,0.2*aper_size*metre_per_pixel*1e6).astype(int))
plt.yticks(ticks=np.arange(0,aper_size,int(aper_size/5)),
           labels=np.arange(0,aper_size*metre_per_pixel*1e6,0.2*aper_size*metre_per_pixel*1e6).astype(int))
plt.xlabel('x (micron)')
plt.ylabel('y (micron)')
plt.imshow(aperture, cmap='gray')
plt.title('Rectangular Aperture')

plt.subplot(1,2,2)
x_cm = x_m*100  # screen position array in cm

# find closest indices to match 5 cm intervals
target_values = []
target_values_pos = np.arange(0, np.max(x_cm), 5)
for i in range(len(target_values_pos)):
  target_values.append(int(-target_values_pos[-i]))
target_values.append(0)
for i in range(len(target_values_pos)):
  target_values.append(int(target_values_pos[i]))
# Create an empty list to store the closest indices
closest_indices = []
for value in target_values:
  # Find the index of the element in x_cm closest to the target value
  closest_index = np.abs(x_cm - value).argmin()
  closest_indices.append(closest_index)

plt.xticks(ticks=closest_indices,labels=target_values,rotation=90)
plt.yticks(ticks=closest_indices,labels=target_values)
plt.title('Diffraction Pattern')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.imshow(inten/np.max(inten), cmap='hot')
plt.clim((0,zlim))
plt.show()
