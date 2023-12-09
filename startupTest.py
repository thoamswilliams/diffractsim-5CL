import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, mm, nm, cm, W, m
red_wl = 632.8
green_wl = 550
dim_l = 15
img_size = 1.5
F = MonochromaticField(
    wavelength= red_wl* nm, extent_x=2*dim_l* mm, extent_y=2*dim_l * mm, Nx=3000, Ny=3000, intensity=0.3*W/(m**2)
)

F.add(ApertureFromImage("geom_exp3.png", image_size=(img_size * mm, img_size * mm), simulation = F))

F.propagate(172*cm)
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-1*dim_l* mm, dim_l* mm], ylim=[-1*dim_l* mm, dim_l* mm])


