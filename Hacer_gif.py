import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
import h5py 
import imageio
#%%
dps, splif = [],[]
spli_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\splines.hdf5'
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp.hdf5'

with h5py.File(spli_hdf, 'r') as f:  
    gspf = f.get('splines_recons')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
        
    for i in range(len(tf_h)):
#        imag.append(im_h[i])
        splif.append([tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]])
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(len(h_dp)):
        dps.append(h_dp[i])
#%%
lista_im = []
t_spl = np.linspace(0,1,1000)
for num in range(1,501,2): #1251
    if (num-1)%20 == 0: print(num,end=' ')
#    ni = '{:04d}'.format(num)
#    im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')
    plt.close('all')
    plt.ioff()
    plt.figure()
    res=1
    if num>=47: res=0
#    xf,yf = splev(t_spl,splif[num-res])
    dpd = dps[num-res]
    lim = 10
    plt.imshow(dpd[lim:-lim,lim:-lim],vmin=-1.6,vmax=0.8)
#    plt.plot(xf, yf, 'r-')
#    plt.set_cmap('gray')
#    plt.colorbar()
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.savefig('imageni.png',bbox_inches='tight')
    lista_im.append(imageio.imread('imageni.png'))
#%%
imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\dat_ftp_2.gif',lista_im,fps=25)
#%%
