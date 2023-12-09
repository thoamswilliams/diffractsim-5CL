import diffractsim
import numpy as np
from tqdm import tqdm
from PIL import Image

diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, mm, nm, cm

def flatten(array):
    out = np.ndarray((array.shape[0], array.shape[1]))

    for i in range(len(array)):
        for j in range(len(array[0])):
            out[i][j] = array[i][j][0] + array[i][j][1] + array[i][j][2]
    
    return out

def mse(A, B):
    return np.square(np.subtract(A, B)).mean()

ref_img = Image.open("150x150_target1.jpg")
ref_img = ref_img.convert("RGB")
ref_img = np.asarray(ref_img)/255
ref_img = flatten(ref_img)
    

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=18 * mm, extent_y=18 * mm, Nx=150, Ny=150
)

apt = ApertureFromImage("150x150black.jpg", image_size=(3 * mm, 3 * mm), simulation = F)

t_arr = apt.get_t()
t_arr_best = t_arr.copy()

best_mse = 2
num_changes = 0
for i in tqdm(range(150)):

    for _ in range(3):
        i = np.random.randint(0, len(t_arr))
        j = np.random.randint(0, len(t_arr[0]))
        t_arr[i][j] = 1 - t_arr[i][j]
    
    F.add_refresh(apt)

    F.propagate_refresh(210*cm)
    rgb = F.get_colors()

    error = mse(flatten(rgb), ref_img)
    print(error)
    if(error < best_mse):
        t_arr_best = t_arr.copy()
        best_mse = error
        num_changes +=1
    else:
        t_arr = t_arr_best.copy()

print(num_changes)
F.plot_colors(rgb, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])
I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm)