import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
from skimage.io import imread
from scipy.interpolate import splev, splprep
from scipy import interpolate
from time import time
#from plantcv import plantcv as pcv
from tqdm import tqdm
import h5py 
#%%
#La función que ordena los splines
def uQuery(pts,u,steps=100,projection=True): 
#https://stackoverflow.com/questions/34941799/querying-points-on-a-3d-spline-at-specific-parametric-values-in-python
    ''' Brute force point query on spline
        pts = [x,y]
        u      = list of queries (0-1)
        steps  = number of curve subdivisions (higher value = more precise result)
        projection = method by wich we get the final result
                     - True : project a query onto closest spline segments.
                              this gives good results but requires a high step count
                     - False: modulates the parametric samples and recomputes new curve with splev.
                              this can give better results with fewer samples.
                              definitely works better (and cheaper) when dealing with b-splines (not in this examples)

    '''
    u = np.clip(u,0,1) # Clip u queries between 0 and 1
    x,y = pts[0],pts[1]
    z = np.zeros_like(x)
    cv = np.vstack((x,y,z)).T
    # Create spline points
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0)
    p = np.array(interpolate.splev(samples,tck)).T  
    # at first i thought that passing my query list to splev instead
    # of np.linspace would do the trick, but apparently not.    

    # Approximate spline length by adding all the segments
    p_= np.diff(p,axis=0) # get distances between segments
    m = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
    s = np.cumsum(m) # cumulative summation of magnitudes
    s/=s[-1] # normalize distances using its total length

    # Find closest index boundaries
    s = np.insert(s,0,0) # prepend with 0 for proper index matching
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) # Find closest lowest boundary position
    i1 = i0+1 # upper boundary will be the next up

    # Return projection on segments for each query
    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    # Else, modulate parametric samples and and pass back to splev
    mod = (((u-s[i0])/(s[i1]-s[i0]))/steps)+samples[i0]
    return np.array(interpolate.splev(mod,tck)).T  

#La función que calcula la curvatura
def curvatura_pap(x,y):
    splin, u = splprep([x,y],s=0)
    spline_1d = splev(u,splin,der=1)
    spline_2d = splev(u,splin,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / \
                 ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return curv

def largo_fib(xf,yf):
    nf = len(xf)
    td = 0
    for i in range(1,nf):
        di = np.sqrt( (xf[i]-xf[i-1])**2 + (yf[i]-yf[i-1])**2  )
        td += di
    return td
#%%
#Traigo splines, y calculo largos
t_spl = np.linspace(0,1,1000)
largos = []
with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
        
    for i in range(len(tf_h)):
        splif = [tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]]
        xf, yf = splev(t_spl, splif)
        lar = largo_fib(xf,yf)
        largos.append(lar)
#%%
# Veo con que fibras quedarme según el largo
xs = np.arange(760,2700)
dele = np.arange(2531,2550) - 760
xs = np.delete(xs,dele)

plt.figure()
plt.plot(xs,largos,'-o')
plt.grid()
plt.show()

largos = np.array(largos)
xsc1 = xs[ (largos<270)]
larc1 = largos[(largos<270)]
xsc = xsc1[larc1>250]
larc = larc1[larc1>250]
    
plt.figure()
plt.plot(xsc,larc,'-o')
plt.grid()
plt.show()
#%%
u = np.linspace(0,1,4000)
steps = 50000

t_spl = np.linspace(0,1,20000)
curm = []

with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
    
    t1 = time()    
    for n, ñ in zip(range(len(tf_h)), tqdm(range(len(tf_h))) ):
        num = xs[n]
        if num in xsc:
            if num < 2531: i = num-760
            if num > 2531: i = num-760-19
            splif = [tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]]

            xf1,yf1 = splev(t_spl,splif)
            xf,yf,z = uQuery([xf1,yf1],u,steps).T
            
            curvf = curvatura_pap(xf,yf)
            curm.append( np.max(curvf) )
                    
    t2 = time()
t2-t1        
#%%
plt.figure()
plt.plot(xsc,curm,'-o')
plt.grid()
plt.show()
curm = np.array(curm)
plt.figure()
plt.plot(xsc[curm<0.01],curm[curm<0.01],'-o')
plt.grid()
plt.show()
#%%
u = np.linspace(0,1,4000)
steps = 50000

t_spl = np.linspace(0,1,20000)

with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
    
    t1 = time()
    num = 780
    if num < 2531: i = num-760
    if num > 2531: i = num-760-19
    splif = [tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]]

    xf1,yf1 = splev(t_spl,splif)
    xf1, yf1 = xf1*0.02086, yf1*0.02086
    xf,yf,z = uQuery([xf1,yf1],u,steps).T
    
    curvf = curvatura_pap(xf,yf)
    t2 = time()
t2-t1
#%%
num = 780 #1376, 2566
if num < 2531: n = num-760
if num > 2531: n = num-760-19
ni = '{:04d}'.format(n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')
    
t_spl = np.linspace(0,1,1000)
xf, yf = splev(t_spl, splif) 
#xf, yf = xf* 0.02086, yf* 0.02086

plt.figure()
plt.imshow(im, extent=(0,1024*0.02086,1024*0.02086,0))
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im, extent=(0,1024*0.02086,1024*0.02086,0))
plt.plot(xf, yf, 'r-')
plt.show()
plt.figure()
plt.plot(u,curvf**2,'-o')
plt.show()
#%%
# Veo de calcular la energia para los que tienen curvatura maxima < 0.01 (sino paso a 0.008)
fibb = xsc[curm<0.008 ] #<0.01]
u = np.linspace(0,1,4000)
steps = 50000

t_spl = np.linspace(0,1,20000)
ener = []
xm, ym = [], []

with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
    
    t1 = time()    
    for n, ñ in zip(range(len(tf_h)), tqdm(range(len(tf_h))) ):
        num = xs[n]
        if num in fibb:
            if num < 2531: i = num-760
            if num > 2531: i = num-760-19
            splif = [tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]]

            xf1,yf1 = splev(t_spl,splif)
            xm.append( np.mean(xf1) )
            ym.append( np.mean(yf1) )
            xf1, yf1 = xf1*0.02086, yf1*0.02086 #paso a cm
            xf,yf,z = uQuery([xf1,yf1],u,steps).T
            
            curvf = curvatura_pap(xf,yf)
            
            ener.append( np.trapz(curvf**2,x=u) )
                    
    t2 = time()
t2-t1
#%%
plt.figure()
plt.plot(fibb,ener,'.-')
plt.show()
plt.figure()
plt.imshow(im)
plt.plot(xm,ym,'ko')
plt.show()
#%%
# Calculo eners por velocidad
arz = np.zeros((1004,1004))
ar = np.arange(1004) 
xx,yy = np.meshgrid( ar,ar ) 
t1 = time()
#corrs = []
vm2_400 = []
with h5py.File('alturas.hdf5', 'r') as f:
    gdp = f.get('hs')
    h_dp = gdp['alts']
    
#    for r in range(25,425,25):
    r = 400
#        vm2 = []
    n = 0
    for i, j in zip(fibb, tqdm(fibb) ):
        dp1 = h_dp[i][10:-10,10:-10]
        dp0 = h_dp[i-1][10:-10,10:-10]
        vel = (250*(dp1-dp0))
        
        pix,piy = int(xm[n]-10),int(ym[n]-10)
        mask = ((xx-pix)**2 + (yy-piy)**2) < r**2
        
        vm2_400.append( np.mean(vel[mask>0]**2) )
        n += 1

#        cor = np.corrcoef(ener,vm2)[0,1]
#        corrs.append(cor)
            
t2=time()
t2-t1
#%%
plt.figure()
#plt.plot(fibb,vm2,'.-')
rs = np.array( list( range(25,425,25) ))
plt.plot(rs ,corrs,'-')
plt.plot(rs ,corrs,'ko')
plt.grid()
plt.xlabel('r (pixeles)')
plt.ylabel('Correlación')
plt.title('Correlación de energías')
plt.show()
#%%
from skimage.filters import gaussian
i = 500
n = fibb[i]#780
print(n)
ni = '{:04d}'.format(n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')


pix,piy = int(xm[i]-10),int(ym[i]-10)

arz = np.zeros_like(im)
ar = np.arange(1024) 
xx,yy = np.meshgrid( ar,ar ) 

#for r in [200,150,100,50]:
r = 200
mask = ((xx-pix)**2 + (yy-piy)**2) < r**2
arz[mask] += 0.1
arz += 0.5

cori = 250
imr = im[piy-cori:piy+cori,pix-cori:pix+cori]
imrm = (im * (arz))[piy-cori:piy+cori,pix-cori:pix+cori]

#plt.figure()
#plt.imshow(im, cmap='gray')
#plt.savefig('imag.png',bbox_inches='tight')
#plt.show()
#plt.figure()
#plt.imshow( imr, extent=(pix-250,pix+250,piy+250,piy-250),  cmap='gray' )
##plt.plot(pix,piy,'ko')
##plt.colorbar()
#plt.savefig('imag_rec.png',bbox_inches='tight')
#plt.show()
plt.figure()
plt.imshow( imrm*3, extent=(pix-250,pix+250,piy+250,piy-250), cmap='gray' )
#plt.plot([689,739],[322,322],'b-',label='r = 50')
#plt.plot([689,787],[322,342],'g-',label='r = 100')
#plt.plot([689,821],[322,392],'y-',label='r = 150')
#plt.plot([689,821],[322,472],'r-',label='r = 200')
#plt.legend()
#plt.text(695,342,'50',color='yellow',fontsize=12)
#plt.text(705,377,'100',color='yellow',fontsize=12)
#plt.text(725,422,'150',color='yellow',fontsize=12)
#plt.text(755,457,'200',color='yellow',fontsize=12)
#plt.colorbar()
#plt.savefig('imag_r'+str(r)+'.png',bbox_inches='tight')
plt.show()

#np.mean( vel[mask>0]**2 )
#%%
fig, ax = plt.subplots(nrows=4,sharex=True,figsize=(10,8))
plt.tight_layout()
ax[0].plot(fibb,np.array(ener),'.-')
#ax[0].set_title('Energía de flexión')
ax[0].set_ylabel(r'$E_f / k_f$')
ax[1].plot(fibb,vm2_50,'r.-',label='r = 50')
#ax[1].set_title('Energía cinética (r = 50)')
ax[1].set_ylabel(r'$E_{\bot} / m$')
ax[1].legend(loc='upper right')#,fontsize=20)
ax[2].plot(fibb,vm2_150,'g.-',label='r = 150')
ax[2].set_ylabel(r'$E_{\bot} / m$')
#ax[2].set_title('Energía cinética (r = 150)')
ax[2].legend(loc='upper right')#,fontsize=20)
ax[3].plot(fibb,vm2_400,'b.-',label='r = 400')
ax[3].set_ylabel(r'$E_{\bot} / m$')
#ax[3].set_title('Energía cinética (r = 400)')
ax[3].legend(loc='upper right')#,fontsize=20)
ax[3].set_xlabel('Frame') #,fontsize=20)
plt.show()
plt.savefig('energias2.png',bbox_inches='tight')
#np.corrcoef(ener,vm2)[0,1]
#%%
#==============================================================================
# Cosas para presentacion
#==============================================================================
t_spl = np.linspace(0,1,20000)

with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
    

    num = 1376#1376, 2566
    if num < 2531: i = num-760
    if num > 2531: i = num-760-19
    splif = [tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]]

    xf,yf = splev(t_spl,splif)

ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

plt.figure()
plt.imshow(im ) 
#plt.plot(xf, yf, 'r-')
plt.xlim(450,650)
plt.ylim(310,0)
plt.savefig('larga.png',bbox_inches='tight')
plt.show()
#%%