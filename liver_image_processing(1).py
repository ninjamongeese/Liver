
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import glob, os

from scipy import misc, ndimage
from scipy.spatial import ConvexHull
import scipy.io as spio


from skimage import filters, morphology,color
from skimage.feature import canny
from skimage.measure import regionprops, label,find_contours
from skimage import exposure, transform, data

from sklearn.cluster import KMeans
plt.rcParams['image.cmap'] = 'viridis' # set colormap to gray by default
plt.rcParams.update({'axes.titlesize': 'x-large'})

from matplotlib_scalebar import scalebar
from matplotlib import ticker


import seaborn
seaborn.set_color_codes()
seaborn.set(style='ticks')


#cylce through line styles
from itertools import cycle
title_size=15

decimation = 1
dpi = 400/decimation
pix_per_mm = dpi/25.4
print(pix_per_mm)
cwd = os.getcwd()
dpmm = 5.2


# # Function Definitions

# In[2]:

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


# In[3]:

def find_coords(img):
    # this find each pin in array wiht val one and return corrds for all vals
    
    coords =[]
    x,y = img.shape
    for j in range(x): # iterates throug each pizel and pulls out the corrdinates afor any one pixle
        for i in range(y):
            if img[j,i]:
                coords.append([i,j])
    return(coords)

def find_array(coords,size):
    #inverse of find coords 
    #given coords, find array where that is true
    
    arr = np.zeros(size)
#     for coord in coords:
#         for x,y in coord: # for each coord, make point true
#             arr[x,y] = 1
    
    for x,y in coords: # for each coord, make point true
        arr[x,y] = 1
    return arr
    


# In[4]:

def find_largest_area(img,num_areas=1):    
    # num is the number of regions we want to leave in image
    label_shapes,num = label(img,return_num=True)  # finds the number of chunks
    area = [] # list of tubles of area and region label

    regions = regionprops(label_shapes) # finds properties about each chunk

    if(num > num_areas):
        n=0    
        for region in regions:
            n=n+1
            for prop in region:
                if prop == 'area':
                    area.append((region[prop],region['label']))
                    #print(n,prop,region[prop])
        area.sort()           
    try:
        cent =regions[area[-1][1]-1]['centroid']  #centroid of largest area?
        bbox = regions[area[-1][1]-1]['bbox'] 
    except:
        print("except")
        cent = regions[0]['centroid']
        bbox = regions[0]['bbox']
    #print(str(num) + ' regions in image')
    #print(area)
    
    #print(cent)
    return cent,bbox,regions,area


# In[6]:

def liver_measure(liver,num_areas = 1):
    # this function is a compilation of all the stuff to do it en masse
    
    #file is the abspulte path to the file
    
    #print(cwd+'/'+file)
    liver = liver[1500:4200-350:decimation,0:3100:decimation]
    
    
    liver_r = liver[:,:,0]
    liver_g =liver[:,:,1]
    liver_b = liver[:,:,2]
    liver_bw = color.rgb2gray(liver)
    liver_lab=color.rgb2lab(liver)
    liver_hsv = color.rgb2hed(liver)
    
    liver_rg = (liver_r>110)/2 + (liver_g>75)+(liver_hsv[:,:,1]>.340)#+(liver_hsv[:,:,0]<-1.1) 
    
    liver_blur = ndimage.gaussian_filter(liver_rg,.135)
    
    n = 3
    shape1 = morphology.disk(n)
    shape2 = morphology.disk(n)
    liver_blur = morphology.erosion(liver_blur,shape1).astype(np.uint8)

    liver_blur = morphology.dilation(liver_blur,shape2).astype(np.uint8)
    liver_filled = ndimage.morphology.binary_fill_holes(liver_blur)
    
    liver_edges = canny(liver_filled,.5)
    
    n = 3
    shape1 = morphology.disk(n)
    shape2 = morphology.disk(n)
    liver_edges = morphology.dilation(liver_edges,shape2).astype(np.uint8)
    liver_edges = morphology.erosion(liver_edges,shape1).astype(np.uint8)

  
    
    shapes = ndimage.morphology.binary_fill_holes(liver_edges)
    cent,bbox,regions,area = find_largest_area(shapes,num_areas)
#     num_areas =1 # num is the number of regions we want to leave in image
#     label_shapes,num = label(shapes,return_num=True)  # finds the number of chunks
#     area = [] # list of tubles of area and region label

#     regions = regionprops(label_shapes) # finds properties about each chunk

#     if(num > num_areas):
#         n=0    
#         for region in regions:
#             n=n+1
#             for prop in region:
#                 if prop == 'area':
#                     area.append((region[prop],region['label']))
#                     #print(n,prop,region[prop])
#         area.sort()           
#     cent =regions[area[-1][1]-1]['centroid']  #centroid of largest area?
#     bbox = regions[area[-1][1]-1]['bbox'] 
#     #print(str(num) + ' regions in image')
#     #print(area)
    
    n_shapes = np.copy(shapes)
    #  set smaller areas to go away
    for a,n in area[:-num_areas]: #area is order list from smalest to largest area
        #print(a,n)
        #next line blocks out other shapes we dont want
        n_shapes[regions[n-1]['bbox'][0]:regions[n-1]['bbox'][2],regions[n-1]['bbox'][1]:regions[n-1]['bbox'][3]] = 0
    #print(len(n_shapes))
    
    #print(cent)
    res = np.zeros_like(liver)
    outline = (1-canny(n_shapes,.5))   
    n = 1
    shape1 = morphology.disk(n)
    outline = morphology.erosion(outline,shape1).astype(np.uint8)
    
    
    for i in range(3):
        res[:,:,i]=liver[:,:,i]*outline
    
    
    
    return 1-outline,regions[area[-1][1]-1],res #return props of shape #


# In[7]:

def overlay_outlines(outlines,properties):
    # This function will take the outlines and overlay them using the centroid
    #shape = ndimage.morphology.binary_fill_holes(1-outline)
    
    #return matrix with outlines shifted to center
    
    new_im = np.zeros_like(outlines[0]) # combination of overlays
    
    for outline in outlines:     
        #err_y = properties[i]['centroid'][0] - 400
        #err_x = properties[i]['centroid'][1] - 600
        #new_im = ndimage.shift(outlines[i],[-err_y,-err_x]) #moves centroid to center
        new_im[outline>0]=1       
        
    return new_im
    


# In[8]:

def center_outlines(outline,properties):
    comb = np.zeros_like(outline)# new overlays
    x,y = comb.shape
    x = np.floor(x/2)
    y = np.floor(y/2)
    err_y = properties['centroid'][0] - x
    err_x = properties['centroid'][1] - y
    new_im = ndimage.shift(outline,[-err_y,-err_x]) #moves centroid to center        
    return new_im
    


# In[9]:

def rotate_image(img,deg,props):
    # this function roatates and image by that amount of degrres
    # positive is ccw, neg is cw
    xM,yM = img.shape
    x = props['centroid'][1]
    y = props['centroid'][0]
    return(transform.rotate(img,deg,preserve_range=True,order=1,
                            resize=False,center= (np.floor(yM/2),np.floor(xM/2)),
                           mode="constant",cval=0.0))



# In[10]:

def avg_outlines(outlines):
    #this function interates througb outlines and finds average of outlines.
    
    #thi function breask up the outlines into 360 degrre chunck for averaging.
    
    for i,outline in enumerate(outlines):
        x_org = 400
        y_org = 600
        for x in outline:
            for y in x:
                if outline[x,y].any():
                    dist = np.linalg.norm(np.array((x,y))-np.array((x_org,y_org)))
                    print(dist)
                    print(x,y)
                    
            


# In[11]:

def plot_overlays(overlays,display=True,save = False):
    #gets list of overalys, and plts and saves them.
    
    
    for overlay,name in overlays:
        plt.imshow(overlay,cmap=plt.cm.gray)
        if display:
            plt.title(name)
            plt.tight_layout()
        if save:
            plt.savefig(name+"_overlay.png",dpi=600)
        if display:
            plt.show()
            
            
def plot_countours(mfill,name,display=True,mm=np.array([False]),save = False,filled=False,
                   rotate='style'):
    #gets list of overalys, and plts and saves them.
    
    low_val = int(np.floor(mfill.min())+1 )
    high_val = int(np.floor(mfill.max())-1  )
    
    percentiles = []
    if (high_val - low_val) > 0 : #
        plt.cla()
        levels = []
        num =high_val - low_val

        #cyle linestyles
        lines = ["-.",":","-","--"]
        linecycler = cycle(lines)
        
        #cyle colors
        color = ['o','m','b',"y"]
        colorcycler = cycle(color)
        
        #print("low:" + str(low_val))
        #print("high:" + str(high_val))
        
        for percentile in [.3*high_val,.5*high_val,.7*high_val]: # find each contour for one image
            #print("lvl")
            contours = find_contours(mfill,level=percentile)
            levels.append(contours)    

        if mm.any(): # imshow min_max outlines
            #print("mm")
            plt.imshow(mm)
        elif filled: #imshow filled shapes
            plt.imshow(mfill)
        else: # imshow nothing (no background)
            plt.imshow(np.zeros_like(mfill))

            
        for level in levels: # for each contour of this value
            for n, contour in enumerate(level): # plot each conout of that value
                
                #find shape of conoutrs and save them
                percent = find_array(contour,mfill.shape)
                percentiles.append(percent)
                
                if rotate == 'style':
                    plt.plot(contour[:, 1], contour[:, 0],next(linecycler),
                             linewidth=.5,color='w' ) # cycle line styles
                elif rotate == 'color':
                    #print("color")
                    plt.plot(contour[:, 1], contour[:, 0],next(colorcycler),
                             linewidth=.5,markersize = .5 ) # cycle line styles
                else:
                    print('Invalide selction for "rotate"')
        plt.axis('off')
        if display:
            plt.title(name)
            plt.tight_layout()
        if save:
            
            plt.savefig(name+"_contours.png",dpi=600)
        if display:
            plt.show()
        
        return percentiles
    else:
        print("no intermediate contours")
        return
    


# In[12]:

def outline_min_max(outlines):
    if(decimation == 1):
        shape2 = morphology.disk(1)
    else:
        shape2 = morphology.disk(1)
    # this function takes the outlines, and finds the outliont that is the most
    # outer edges and the most inner edges
    max_comb = np.zeros_like(outlines[0])
    min_comb = np.zeros_like(outlines[0])
    
    
    for outline in outlines:  
        new = outline
        
#         plt.imshow(new)
#         plt.show()
        
#         if(decimation == 1):
#             new = morphology.dilation(new,shape2)
#             new = morphology.dilation(new,shape2)
#         plt.imshow(new)
#         plt.show()
        
        filled = ndimage.morphology.binary_fill_holes(new)
#         plt.imshow(filled)
#         plt.show()
        max_comb = max_comb + filled
        
    max_outline = canny(max_comb>0,.1)
    min_outline = canny(max_comb > (max_comb.max()-1),.1)
    
    
    min_outline = morphology.dilation(min_outline,shape2).astype(np.uint8)
    max_outline = morphology.dilation(max_outline,shape2).astype(np.uint8)

    return max_outline,min_outline,max_comb
        


# # Read in Files
# 
# This next chunck reads in files and keeps track of the names

# In[61]:

# this explores the scans folder and finds the paths to the images im interested in

files = list_files(cwd + r"/scans")
names = []
experiments = []
for file_pth in files:
    if "YZ" in file_pth:
        continue #skip the YZ cuts for now.
    if 'edges' in file_pth:
        continue #skip photes edited
    
    folder = file_pth.replace(cwd+ r"/scans",'')[1:]
    #print(name)
    i = folder.find("\\")
    if (i>0):
        name2 = folder[:i]
        if name2 not in names:
            print(name2)
            names.append(name2)
            experiments.append([name2,[]])
    else:
        pass
    for name in names:
        if name in file_pth:
            experiments[-1][1].append(file_pth)
    print(file_pth)



# ## Image processing
# 
# Now we can use the files and actaully do some image processingon them 
# 
# THis next section is the meat of the program. This is where we actaully find the ablation outlines and then start with esgemetnign every thing out

# In[62]:

overlays = []
outlines = []
centered = []
names=[]

twenty_p = []
twenty_in = []
fifteen_p = []
fifteen_in = []

twenty_p_props = []
twenty_in_props = []
fifteen_p_props = []
fifteen_in_props = []

rotations = [  #nameof image to rotate and amount in degrees(+ = ccw)
    ['img035', 20],
    ['img034', -5],
    ['img033', 10],
    ['img032', -20],
    ['img031', -10],
    ['img030', 20],
    ['img029',50],
    ['img025',13],
    ['img021',5],
    ['img019',10],
    ['img018',15],
    ['img016',10],
    
    ['img012',-22],
    ['img008',-15],
    ['img006',0],
    
    
]
for experiment in experiments: # each setup
    regions = []
    outlines.append([experiment[0],[]])
    centered.append([experiment[0],[]])
    names.append(experiment[0])
    for trial in experiment[1]: #each time I ran a setup
        print(trial)
        liver = misc.imread(trial) # read in image
        plt.cla()

        outline,region,res = liver_measure(liver) # find outline of biggest ablation
        plt.imshow(res)
        plt.grid("off")
        plt.plot(region['centroid'][1],region['centroid'][0],'ro',linewidth=10)
        plt.text(500,100, r'Area($ \mathrm{mm}^2 $):  ' + str(region["Area"]/(pix_per_mm**2)),color='w')
        plt.savefig('edges_'+trial[-10:],dpi=800,bbox_inches='tight')
        
        cent = center_outlines(outline,region) #put outlines in same spot
   
        for rotation in rotations: # roate those pciture that need it
            if rotation[0] in trial:
                
                
                #uncomment nexxt line only
                cent = rotate_image(cent,rotation[1],region) 
                print('roatate image')
                plt.imshow(cent)
                #plt.show()
                

        
        
        #save filled countours 
        #names: ['15 to 10', '15mm Parallel', '20 to 15', '20mm Parallel']
        if(names[-1] == '15 to 10'):
            filled =ndimage.binary_fill_holes(cent)
            
            fifteen_in.append(filled)
            fifteen_in_props.append(region)
            
            
        elif(names[-1] == '15mm Parallel'):
            filled =ndimage.binary_fill_holes(cent)
            
            fifteen_p_props.append(region)
            fifteen_p.append(filled)
            
        elif (names[-1] == '20 to 15'):
            filled =ndimage.binary_fill_holes(cent)
            
            twenty_in_props.append(region)
            twenty_in.append(filled)
            
        elif(names[-1] == '20mm Parallel'):
            filled =ndimage.binary_fill_holes(cent)
            
            twenty_p_props.append(region)
            twenty_p.append(filled)
            
        else:
            print("err")
        
        plt.cla()
        plt.imshow(cent)
        plt.axis('off')
        plt.savefig('outline_'+trial[-10:],dpi=800,bbox_inches='tight')
        #plt.show()
        
        centered[-1][1].append(cent)
        outlines[-1][1].append(outline)
        regions.append(region)
        print("Area: ")
        print(region["Area"]*pix_per_mm)
        
    overlay=overlay_outlines(centered[-1][1],regions) #create an overlay of each trial for this experiment
    overlays.append([overlay,experiment[0]])
    
print("done!")
    


# In[63]:

#plot_overlays(overlays,save=True)


# In[64]:

contour =0 
setups=[]
exp_15mm_p = []
exp_15mm_TipDist10 = []
exp_20mm_p = []
exp_20mm_TipDist15 = []

exp_Tmap15=[]
exp_Tmap20=[]

for center in centered: #for each centered ablation outlin
    perc = [] #holds consective outline of percentage ablated
    
    max_outline,min_outline,mfill = outline_min_max(center[1]) # find the most outside and inside percentile conoutours
    perc.append(max_outline)
    perc.append(min_outline) 
    
    plt.imshow(mfill)
    plt.tight_layout()
    plt.axis('off')
    #scaleBar = scalebar.ScaleBar(1000*(1/pix_per_mm) ,'mm') # 1 pixel = 1/2 mm
    #plt.gca().add_artist(scaleBar)
    plt.savefig('mfill_'+center[0]+'.png',dpi=600)
    
    x,y = max_outline.shape
    print(x,y)
    
    min_max = np.zeros((x,y,3))# put min and mox percentile contours into one RGB  image/array
    min_max[:,:,0] = max_outline
    min_max[:,:,1] = min_outline
    
    s = min_max.sum(axis=2)
    print("S shape: " + str(s.shape))
    min_max[s==0]=1
                
    print(max_outline.shape)
    print(min_max.max())
    
    plt.cla()
    plt.imshow(min_max)
    plt.xlim(950, 2100)
    plt.ylim(1750,500)
    plt.tight_layout()
    scaleBar = scalebar.ScaleBar((1/pix_per_mm) ,'mm') # 1 pixel = 1/2 mm
    plt.gca().add_artist(scaleBar)
    plt.axis('off')
    plt.title(center[0],fontsize=title_size)
    plt.savefig('min_max_test_'+center[0]+'.png',dpi=600)
    
    #plt.show()
    plt.cla()
    
    print("contour")
    
    if('15mm Parallel' in center[0] or "15 to 10" in center[0]):
        exp_Tmap15.append(mfill)
        print("append 15")
    if('20mm Parallel' in center[0] or "20 to 15" in center[0]):
        exp_Tmap20.append(mfill)
        print("append 20")
    
    
    
#     #find percentile contours and save them
#     perc_cont = plot_countours(mfill,center[0],mm=min_max,save=True,display=True,rotate='color')
#     if perc_cont:
#         perc = perc + perc_cont #adds intermediate percentage outlines
#     setups.append(perc)
    
    
    
    
    #contours = find_contours(mfill,level=1)
#     #min_max_cont = min_max
#     #min_max_cont[:,:,2] = find_array(contours,mfill.shape)
#     plt.imshow(min_max_cont)
#     for n, contour in enumerate(contours):
#         plt.plot(contour[:, 1], contour[:, 0], linewidth=1)

#     plt.savefig('min_max_countour_'+center[0]+'.png',dpi=600)
#     plt.show()
#     plt.cla()

#print(max(max(max_outline.any())))

np.savez('Tmaps.npz',exp_Tmap15=exp_Tmap15,exp_Tmap20=exp_Tmap20)
print("done")


# In[65]:

# This is all dealing with the percentile contours. Not the actaull ablation conours themselves

small = []

bin_maps = []
props_perc = []

for setup in setups:
    areas = []
    if setup:
        bin_maps.append([])
        props_perc.append([])
        #print("new setup")
        for percentile in setup:
            area= []
            n=0
            new_img = ndimage.binary_fill_holes(percentile)
            labels = label(new_img)
            regions = regionprops(labels)
            for region in regions:
                for prop in region:
                    if prop == "area":
                        area.append([region['area'],region["label"]])
                        
            #print(new_img.shape)
            
            #print(area)          
            
            if len(area) > 1:  #deleate all the small random areas
                area.sort()
                #print(area)
                new_img[labels !=area[-1][1]] = 0
                
            #print(new_img.shape)
            
            if area[-1][0]>1000:
                #print(new_img.shape)
                #plt.imshow(new_img)
                #plt.show()
                bin_maps[-1].append(new_img)
                props_perc[-1].append(regions[area[-1][1]-1])
                #print(props_perc[-1][-1]['area'])
                areas.append(props_perc[-1][-1]['area'])
        #sort by area
        zipped = list(list(zip(areas,props_perc[-1],bin_maps[-1])))
        zipped.sort()
        #print(zipped)
        areas,props_perc[-1],bin_maps[-1] = zip(*zipped)


# In[66]:

twenty_p_percent = []
twenty_in_percent = []
fifteen_p_percent = []
fifteen_in_percent = []

for n,experiment in enumerate(bin_maps):
    print(n)
    for trial in experiment:
        if(n ==0):
            fifteen_in_percent.append(trial)
        elif(n == 1):
            fifteen_p_percent.append(trial)
        elif (n==2):
            twenty_in_percent.append(trial)
        elif(n ==3):
            twenty_p_percent.append(trial)
        else:
            print("err")
print("done")


# In[70]:

# calculates the average area of each ablation in cm

area= 0

for region in twenty_in_props:
    for prop in region:
        if prop =='area':
            print(prop,region[prop]/pix_per_mm)
            area = area + region[prop]
average_area = area/len(twenty_in_props)
print(average_area/pix_per_mm)

area= 0            
for region in twenty_p_props:
    for prop in region:
        if prop =='area':
            print(prop,region[prop]/pix_per_mm)
            area = area + region[prop]
average_area = area/len(twenty_in_props)
print(average_area/pix_per_mm)

area= 0            
for region in fifteen_in_props:
    for prop in region:
        if prop =='area':
            print(prop,region[prop]/pix_per_mm)
            area = area + region[prop]
average_area = area/len(fifteen_in_props)

print(average_area/pix_per_mm)

area= 0           
for region in fifteen_p_props:
    for prop in region:
        if prop =='area':
            print(prop,region[prop]/pix_per_mm)
            area = area + region[prop]
average_area = area/len(fifteen_p_props)
print(average_area/pix_per_mm)


# In[71]:

#print areas of percentile curves
for setup in props_perc:
    print("new")
    for prop in setup:
        print(prop["area"]/pix_per_mm)
        


# # Load  Simulation Data

# In[30]:

pth = r"C:\\Users\\ninja\\Dropbox\\Research\\Praksh\\tall\\"
pth = r"C:\Users\awhite64\Dropbox\Research\Praksh\tall\\"
matlab_files = [
    r'10mm_p_lp_tall.mat',
    r"10mm_TipDist5_lp_tall.mat",
    r"15mm_p_lp_tall.mat",
    r"15mm_TipDist10_lp_tall.mat",
    r"15mm_TipDist5_lp_tall.mat",
    r"20mm_p_lp_tall.mat",
    r"20mm_TipDist15_lp_tall.mat",
    r"20mm_TipDist10_lp_tall.mat",
    r"20mm_TipDist5_lp_tall.mat"
]

# create plots for indivuduale heat maps
# populate as more become avaible
Tmaps20 = [None,None,None,None]
Tmaps15 = [None,None,None]
Tmaps10 = [None,None]
plt.cla()
for m in matlab_files:
    f= spio.loadmat(pth+m)
    #print(f)
    print(m)
    i = m.rfind("\\")+1
    ant_spaceing = str(m[i:i+2])
    print("ant spavceing:" + ant_spaceing + "!")
    print(i)
    for k in f.keys():        
        print(k)
        if "TipDist" in k:
            print("non parallel")
            if ant_spaceing == '20':
                if "TipDist15"in k:
                    Tmaps20[1] = f["Tmap20_TipDist15"]
                elif "TipDist10" in  k:
                    Tmaps20[2] = f["Tmap20_TipDist10"]
                elif "TipDist5" in  k:
                    Tmaps20[3] = f["Tmap20_TipDist5"]
            
            elif ant_spaceing == '15':
                print("15")
                if "TipDist10" in  k:
                    Tmaps15[1] = f["Tmap15_TipDist10"]
                elif "TipDist5" in  k:
                    
                    
                    Tmaps15[2] = f["Tmap15_TipDist5"]
            
            elif ant_spaceing == '10':
                if "TipDist5" in  k:
                    Tmaps10[1] =  f["Tmap10_TipDist5"]
            
        elif "_p_" in k:
            #PARALLEL CASES
            if '10' in k:
                Tmaps10[0] = f['Tmap10_p_tall']
            elif '15' in k:
                Tmaps15[0] = f['Tmap15_p_tall']
            elif '20' in k:
                Tmaps20[0] = f["Tmap20_p_tall"]

labels=["Parallel","5mm in","10mm in","15mm in"]
colors = ['orange',"yellow",'white','red']
colorcycler = cycle(colors)
plt.cla()
#spaceing = '20' 
for spaceing,scenario in zip(["10","15","20"],[Tmaps10,Tmaps15,Tmaps20]):
    for i,Tmap in enumerate(scenario):
        if Tmap is None:#if Tmap exists
            continue
        #find name for title/Pic name
        if i ==0: #  Paralle case
            name = spaceing+"mm Parallel"
            p_name = name
        else: #non parallel case
            name = spaceing + "mm to " + str(int(spaceing)-i*5) +"mm"

        print(name)

        #plot XZ plane
        xzPlane = np.rot90(Tmap[101,:,:])
        cs = plt.contour(xzPlane,levels = [55],colors='white')
        plt.clabel(cs,fmt='%.0f', inline=True)
        plt.imshow(xzPlane,aspect=2,cmap=plt.cm.viridis) #not sure about the aspect ratio
        plt.title(name + ' XZ',fontsize=20)
        plt.xlim(40, 160)
        plt.ylim(120,40)
        plt.axis('off')
        cb = plt.colorbar()
        cb.ax.set_title('°C')
        
        scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5)
        plt.gca().add_artist(scaleBar)
        
        
        #plt.tight_layout()
        plt.savefig(name + ' XZ'+".png",dpi = 800)
        plt.show()
        plt.cla()

        #plot YZ plane
        yzPlane = Tmap[:,:,75]
        cs = plt.contour(yzPlane,levels=[55],colors='white')
        plt.clabel(cs,fmt='%.0f', inline=True)
    
        plt.imshow(yzPlane,aspect=1,cmap=plt.cm.viridis) #not sure about the aspect ratio
        cb = plt.colorbar()
        cb.ax.set_title('°C')
        scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5)
        plt.gca().add_artist(scaleBar)
        plt.axis('off')
        plt.xlim(40, 160)
        plt.ylim(160, 40)
        plt.title(name + ' YZ',fontsize=20)
        plt.tight_layout()
        
        plt.savefig(name + ' YZ'+".png",dpi = 800)        
        plt.show()
        plt.cla()
        
        #plot XY plane
        xyPlane = np.rot90(Tmap[:,101,:])
        cs = plt.contour(xyPlane,levels=[55],colors='white')
        plt.clabel(cs,fmt='%.0f', inline=True)
        plt.imshow(xyPlane,aspect=2,cmap=plt.cm.viridis) #not sure about the aspect ratio
        scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5)
        plt.gca().add_artist(scaleBar)
        plt.axis('off')
        plt.title(name + ' XY',fontsize=20)
        plt.xlim(40, 160)
        plt.ylim(120, 40)
        cb = plt.colorbar()
        cb.ax.set_title('°C')
        plt.tight_layout()
        plt.savefig(name + ' XY'+".png",dpi = 800)   
        plt.show()
        
        plt.cla()
    
   
# # for file in matlab_files:
# #     i = file.rfind(r'thesis')
# #     name = file[i+12:]
# #     if "15mm_p_lp_tall" in name:
# #         sim_15mm_p_lp_tall = spio.loadmat(file,squeeze_me=True);
# #     elif "15mm_TipDist10_lp_tall" in name:
# #         #sim_15mm_TipDist10_lp_tall.mat = spio.loadmat(file,squeeze_me=True);
# #     elif "20mm_p_lp_tall" in name:
# #         #sim_20mm_p_lp_tall = spio.loadmat(file,squeeze_me=True);
# #     elif "20mm_TipDist15_lp_tall" in name:
# #         #sim_20mm_TipDist15_lp_tall = spio.loadmat(file,squeeze_me=True);
        
# #plt.cla()
# plt.imshow(f,aspect=2,cmap=plt.cm.viridis) #not sure about the aspect ratio
# plt.contour(f,levels=[55],colors='white')
# plt.xlim(15, 135)
# plt.ylim(135,65)
# plt.axis('off')
# scaleBar = scalebar.ScaleBar(.5,'mm') # 1 pixel = 1/2 mm
# plt.gca().add_artist(scaleBar)
# plt.title('15mm Parallel',fontsize=title_size)
# plt.savefig('15mm_Parallel.png',dpi=600)
# plt.show()

print("done")


# In[18]:

## overlay plots
labels=["Parallel","5mm in","10mm in","15mm in"]
colors = ['white',"gold","yellowgreen",'black']
colorcycler = cycle(colors)
plt.cla()
#spaceing = '20' 
for spaceing,scenario in zip(["10","15","20"],[Tmaps10,Tmaps15,Tmaps20]):
    # overlay plots
    colorcycler = cycle(colors)
    
    #XZ Overlay
    for i,Tmap in enumerate(scenario):
        if Tmap is None:#if Tmap exists
            continue
        #find name for title/Pic name
        if i ==0: #  Paralle case
            name = spaceing+"mm Parallel"
            p_name = name
        else: #non parallel case
            name = spaceing + "mm to " + str(int(spaceing)-i*5) +"mm"

        xzPlane = np.rot90(Tmap[101,:,:])
        cs = plt.contour(xzPlane,levels = [55],colors = next(colorcycler))
        cs.collections[0].set_label(labels[i])
        #i=i+1
    colorcycler = cycle(colors)
    plt.imshow(np.rot90(scenario[0][101,:,:]),aspect=2,cmap=plt.cm.viridis)
    plt.xlim(40, 160)
    plt.ylim(120,40)
    plt.axis('off')
    scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5,location='lower right')
    plt.gca().add_artist(scaleBar)
    cb = plt.colorbar()
    cb.ax.set_title('°C')
    plt.legend(loc='lower left',frameon=True)
    plt.title(p_name + ' XZ Overlay',fontsize=20)
    plt.tight_layout()
    plt.savefig(name + ' XZ Overlay'+".png",dpi = 800)
    plt.show()
    plt.cla()
    
    #YZ Overlay
    #plane = [:,:,75]
    colorcycler = cycle(colors)
    for i,Tmap in enumerate(scenario):
        if Tmap is None:#if Tmap exists
            continue
        #find name for title/Pic name
        if i ==0: #  Paralle case
            name = spaceing+"mm Parallel"
            p_name = name
        else: #non parallel case
            name = spaceing + "mm to " + str(int(spaceing)-i*5) +"mm"

            
        yzPlane = np.rot90(Tmap[:,:,75])
        cs = plt.contour(yzPlane,levels = [55],colors = next(colorcycler))
        cs.collections[0].set_label(labels[i])
        
        #i=i+1
    plt.gca().set_color_cycle(None)
    plt.imshow(np.rot90(scenario[0][:,:,75]),aspect=2,cmap=plt.cm.viridis)
    plt.xlim(40, 160)
    plt.ylim(160, 40)
    plt.axis('off')
    scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5,location='lower right')
    plt.gca().add_artist(scaleBar)
    cb = plt.colorbar()
    cb.ax.set_title('°C')
    plt.legend(loc='lower left',frameon=True)
    plt.title(p_name + ' YZ Overlay',fontsize=20)
    plt.tight_layout()
    plt.savefig(name + ' YZ Overlay'+".png",dpi = 800)
    plt.show()
    plt.cla()

    #XY Overlay
    #plane = [:,101,:]
    colorcycler = cycle(colors)
    for i,Tmap in enumerate(scenario):
        if Tmap is None:#if Tmap exists
            continue
        #find name for title/Pic name
        if i ==0: #  Paralle case
            name = spaceing+"mm Parallel"
            p_name = name
        else: #non parallel case
            name = spaceing + "mm to " + str(int(spaceing)-i*5) +"mm"

        yzPlane = np.rot90(Tmap[:,101,:])
        cs = plt.contour(yzPlane,levels = [55],colors = next(colorcycler))
        cs.collections[0].set_label(labels[i])
        #i=i+1
    plt.gca().set_color_cycle(None)
    plt.imshow(np.rot90(scenario[0][:,101,:]),aspect=2,cmap=plt.cm.viridis)
    plt.xlim(40, 160)
    plt.ylim(120, 40)
    plt.axis('off')
    scaleBar = scalebar.ScaleBar(.5,'mm',frameon=True,box_alpha=.8,pad = .5,border_pad=.5,location='lower right')
    plt.gca().add_artist(scaleBar)
    cb = plt.colorbar()
    cb.ax.set_title('°C')
    plt.legend(loc='lower left',frameon=True)
    plt.title(p_name + ' XY Overlay',fontsize=20)
    plt.tight_layout()
    plt.savefig(name + ' XY Overlay'+".png",dpi = 800)
    plt.show()
    plt.cla()


# In[77]:

exp_Tmap20


# ## create overalys of image precessing areas

# In[78]:

# load mfill objects
data = np.load("Tmaps.npz")
print(data.keys())
exp_Tmap15 = data['exp_Tmap15']
exp_Tmap20 = data['exp_Tmap20']

data = np.load("Tmaps.npz")
print(data.keys())


#color = ['#C6CB1A',"#CB271A"]
colors = ['steelblue',"lightcoral"]
#lstyle = ['dashed','solid',"dashdot","dotted"]
lstyle = ['dashed','solid']

colorcycler = cycle(colors)
stylecycler = cycle(lstyle)
labels = ['Non Parallel',"Parallel"]
#lvls = [0,4]
lvls = [5*.75]
i=0
for t in exp_Tmap15:
    cs = plt.contour(t,levels=lvls,colors=next(colorcycler),linestyles=next(stylecycler),linewidths = 4)
    #plt.clabel(cs,fmt='%.1f', inline=False)
    cs.collections[0].set_label(labels[i])
    i=i+1
    
#plt.imshow(exp_Tmap15[1],cmap=plt.cm.gray)
plt.axes().set_aspect('equal')
plt.grid("off")    
plt.axis("off")
plt.xlim(1150, 1950)
plt.ylim(1600,700)
scaleBar = scalebar.ScaleBar((1/pix_per_mm) ,'mm') # 1 pixel = 1/2 mm
plt.gca().add_artist(scaleBar)

plt.title("15mm 75% Ablation Overlay",fontsize=title_size)
plt.legend(loc='upper left',prop={'size': 12})
plt.tight_layout()
plt.savefig("Exp_15mm_overlay.png", dpi=800)
plt.show()

i=0
for t in exp_Tmap20:
    cs = plt.contour(t,levels=lvls,colors=next(colorcycler),linestyles=next(stylecycler),linewidths = 4)
    #plt.clabel(cs,fmt='%.1f', inline=False)
    cs.collections[0].set_label(labels[i])
    i=i+1

    
#plt.imshow(exp_Tmap15[1],cmap=plt.cm.gray)
plt.axes().set_aspect('equal')
plt.grid("off")    
plt.axis("off")
plt.xlim(1150, 1950)
plt.ylim(1600,700)
scaleBar = scalebar.ScaleBar((1/pix_per_mm) ,'mm') # 1 pixel = 1/2 mm
plt.gca().add_artist(scaleBar)

plt.title("20mm 75% Ablation Overlay",fontsize=title_size)
plt.legend(loc='upper left',prop={'size': 12})
plt.tight_layout()
plt.savefig("Exp_20mm_overlay.png", dpi=800)
plt.show()


# # DSC calculations
# 
# We can calulate how simialr tow areas or vlumes ar by iuing the dice similarty coeeficnets. 
# 
# $DSC = \frac{|A \cup B\ |}{A \cap B}$

# In[67]:

DSC_20 = []
for exp in twenty_p:
    A = exp
    for exp2 in twenty_in:
        B = exp2
#         plt.imshow(A)
#         plt.show()
#         plt.imshow(B)
#         plt.show()
        J_stat = sum(sum(A & B))/sum(sum(A | B))
        dsc = 2*J_stat/(1+J_stat)
        print(dsc)
        DSC_20.append(dsc)
        
print("Mean: " + str(np.mean(DSC_20)))
print("var: " + str(np.var(DSC_20)))
print("STD: " + str(np.std(DSC_20)))


# In[68]:

#DSC 
DSC_l5 = []
#compare DSC of two experiments, each ablation agains the others        
for exp1 in fifteen_in:
    A = exp1
    for exp2 in fifteen_p:
        B = exp2
        
        num = sum(sum(A[:] & B[:]))
        den = sum(sum(A[:] | B[:]))
        #print(num)
        #print(den)
        J_stat = num/den
        #print(J_stat)
        DSC = 2*J_stat/(1+J_stat) 
        print(DSC)
        DSC_l5.append(DSC)
        
        
        
print("Mean: " + str(np.mean(DSC_l5)))
print("var: " + str(np.var(DSC_l5)))
print("std_dev " + str(np.sqrt(np.var(DSC_l5))))


# In[16]:

# volumetirc DSC of simulated Tmaps
print("20---------")
for i,tmapA in enumerate(Tmaps20[:-1]):
    A = tmapA > 55
    print(labels[i])
    for tmapB in Tmaps20[i:]:
        B = tmapB > 55
        num = sum(sum(sum(A[:] & B[:])))
        den = sum(sum(sum(A[:] | B[:])))
        J_stat = num/den
        DSC = 2*J_stat/(1+J_stat) 
        print(DSC)
        #DSC_l5.append(DSC)

print("15--------")
for i,tmapA in enumerate(Tmaps15[:-1]):
    A = tmapA > 55
    print(labels[i])
    for tmapB in Tmaps15[i:]:
        B = tmapB > 55
        num = sum(sum(sum(A[:] & B[:])))
        den = sum(sum(sum(A[:] | B[:])))
        J_stat = num/den
        DSC = 2*J_stat/(1+J_stat) 
        print(DSC)
        #DSC_l5.append(DSC)
print("10-------")        
for i,tmapA in enumerate(Tmaps10[:-1]):
    A = tmapA > 55
    print(labels[i])
    for tmapB in Tmaps10[i:]:
        B = tmapB > 55
        num = sum(sum(sum(A[:] & B[:])))
        den = sum(sum(sum(A[:] | B[:])))
        J_stat = num/den
        DSC = 2*J_stat/(1+J_stat) 
        print(DSC)
        #DSC_l5.append(DSC)
        


# # Testing below

# In[4]:


liver = misc.imread(cwd + r"\scans_test1\img010.jpg")
liver = liver[1500:4200-350:decimation,0:3100:decimation] #crop out other stuff for testing img010
# sampled every third point to avoid memory errors
plt.imshow(liver)
plt.grid("off")
plt.show()
print(liver.shape)


# # Image Segmentation
# Try to segment out the part of the lvier that is cooked
# 
# To do this I transformed the image into varius color schemes(fromg RGB to HSV andLAB) and used a threshold on it, then recombined the most usefull ones
# 

# In[5]:

# test rgb cmap

liver_r = liver[:,:,0]
liver_g =liver[:,:,1]
liver_b = liver[:,:,2]
liver_bw = color.rgb2gray(liver)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows = 1, ncols=4,figsize=(10,4),
                                sharex=False,sharey=False)

ax1.imshow(liver_r>100,cmap=plt.cm.gray)
ax1.axis("off")
ax1.set_title("Red", fontsize = 20)

ax2.imshow(liver_g>60, cmap=plt.cm.gray)
ax2.axis("off")
ax2.set_title('Green', fontsize = 20)

ax3.imshow(liver_b>50, cmap=plt.cm.gray)
ax3.axis("off")
ax3.set_title('BLue', fontsize =20)

ax4.imshow(liver_bw>.24, cmap=plt.cm.gray)
ax4.axis("off")
ax4.set_title('B+W', fontsize =20)
plt.show()


# The picture labled "Red" above looks like it has a decent representation of the cooked part of the liver. What that one is missing is a little divit in the left side of the left peice of liver

# In[6]:

# test lab cmap

liver_lab=color.rgb2lab(liver)
fig,(ax1,ax2,ax3) = plt.subplots(nrows = 1, ncols=3,figsize=(8,3),
                                sharex=True,sharey=True)

ax1.imshow(liver_lab[:,:,0]>30,cmap=plt.cm.gray)
ax1.axis("off")
ax1.set_title("Red", fontsize = 20)

ax2.imshow(liver_lab[:,:,1]>15, cmap=plt.cm.gray)
ax2.axis("off")
ax2.set_title('Green', fontsize = 20)

ax3.imshow(liver_lab[:,:,2]>13, cmap=plt.cm.gray)
ax3.axis("off")
ax3.set_title('Blue', fontsize =20)
plt.show()


# The picture labled "Blue" above is also a nice representation, and it has the divt filed in. It might be a little more usefull

# In[7]:

# test hsv cmap
liver_hsv = color.rgb2hed(liver)
fig,(ax1,ax2,ax3) = plt.subplots(nrows = 1, ncols=3,figsize=(8,3),
                                sharex=True,sharey=True)

ax1.imshow(liver_hsv[:,:,0]<-1.1,cmap=plt.cm.gray)
ax1.axis("off")
ax1.set_title("Red", fontsize = 20)

ax2.imshow(liver_hsv[:,:,1]>.342, cmap=plt.cm.gray)
ax2.axis("off")
ax2.set_title('Green', fontsize = 20)

ax3.imshow(liver_hsv[:,:,2], cmap=plt.cm.gray)
ax3.axis("off")
ax3.set_title('Blue', fontsize =20)

plt.show()


# The pictures "Red" and "Green" are both ok, I think the green is better since it's more selective, the Red looks really large
# 
# Now we can try combining the selected images and see what the total result looks like

# In[8]:

# test hsv cmap
liver_hsv = color.rgb2hed(liver)
# fig,(ax1,ax2,ax3) = plt.subplots(nrows = 1, ncols=3,figsize=(20,5),
#                                 sharex=True,sharey=True)

plt.imshow((liver_r>110),cmap=plt.cm.gray)
plt.axis("off")
plt.title("Red", fontsize = 25)
plt.savefig("red_thresh",dpi = 800)
plt.show()

plt.imshow((liver_g>75), cmap=plt.cm.gray)
plt.axis("off")
plt.title('Green', fontsize = 25)
plt.savefig("green_thresh",dpi = 800)
plt.show()

plt.imshow(liver_hsv[:,:,1]>.340, cmap=plt.cm.gray)
plt.axis("off")
plt.title('Hematoxylin', fontsize =25)
plt.tight_layout()
plt.savefig('Hematoxylin_thresh.png',dpi=800)
plt.show()


# In[9]:

liver_rg = (liver_r>110)/2 + (liver_g>75)+(liver_hsv[:,:,1]>.340)#+(liver_hsv[:,:,0]<-1.1) 
liver_rg = liver_rg>0
plt.imshow(liver_rg>.01,cmap=plt.cm.gray)
plt.title('Combined Binary Images', fontsize = 25)
plt.axis("off")
plt.savefig("Color_Res.png",dpi = 800)
plt.show()


# In[23]:

res = np.zeros_like(liver)
for i in range(3):
    res[:,:,i]=liver[:,:,i]*(n_shapes)



fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(res)
#plt.axes("off")
plt.savefig('Segmented_final.png',dpi=800)
plt.tight_layout()
plt.show()


# In[11]:

#liver_edge = canny(ndimage.gaussian_filter(liver_bw,2),.25)

blur = .135*np.sqrt(2)
liver_blur = ndimage.gaussian_filter(liver_rg,blur)
#liver_blur = ndimage.gaussian_filter(liver_blur,blur)

#liver_blur = liver_rg
plt.imshow(liver_blur>0,cmap=plt.cm.gray)
plt.axis("off")
plt.title("Gaussion Blurred $\sigma = .19$",fontsize = 20)
plt.savefig("gaussion_blur.png",dpi=800)
plt.show()


# In[12]:

n = 2
shape1 = morphology.disk(n)
shape2 = morphology.disk(n)
t = morphology.erosion(liver_blur,shape1).astype(np.uint8)

t = morphology.dilation(t,shape2).astype(np.uint8)

plt.imshow(t,cmap=plt.cm.gray)
plt.axis("off")
#plt.title("Morphological Operators", fontsize = 20)
plt.savefig("Morphed_img.png", dpi = 800)
plt.show()


# In[13]:

liver_filled = ndimage.morphology.binary_fill_holes(t)
#liver_filled = ndimage.gaussian_filter(liver_filled,1.2)
plt.imshow(liver_filled,cmap=plt.cm.gray)
plt.axis("off")
plt.savefig("filled_wholes.png",dpi=800)
plt.show()


# In[14]:

liver_edges = canny(liver_filled,.5)
plt.imshow(liver_edges,cmap=plt.cm.gray)
plt.show()


# In[15]:

n = 3
shape1 = morphology.disk(n)
shape2 = morphology.disk(n)
liver_edges = morphology.dilation(liver_edges,shape2).astype(np.uint8)
liver_edges = morphology.erosion(liver_edges,shape1).astype(np.uint8)


plt.imshow(color.rgb2gray(liver)*(1-liver_edges),cmap=plt.cm.gray)
plt.show()


# In[16]:

liver_edges.max()


# In[17]:

#liver_edges = abs(1-liver_edges)
shapes = ndimage.morphology.binary_fill_holes(liver_edges)
plt.imshow(shapes,cmap=plt.cm.gray)
plt.show()


# # Area selection
# Now that we have the image divied up into discret chunks, we are going to choose to keep the two larges ones. Hopefully the cooked liver ones

# In[18]:

#shapes = liver_filled
num_areas =1 # num is the number of regions we want to leave in image
label_shapes,num = label(shapes,return_num=True)  # finds the number of chunks
area = [] # list of tubles of area and region label

regions = regionprops(label_shapes) # finds properties about each chunk

if(num > num_areas):
    n=0    
    for region in regions:
        n=n+1
        for prop in region:
            if prop == 'area':
                area.append((region[prop],region['label']))
                print(n,prop,region[prop])
    area.sort()           

print(str(num) + ' regions in image')
print(area)

n_shapes = np.copy(shapes)
#  set smaller areas to go away
for a,n in area[:-num_areas]: #area is order list from smalest to largest area
    print(a,n)
    #next line blocks out other shapes we dont want
    n_shapes[regions[n-1]['bbox'][0]:regions[n-1]['bbox'][2],regions[n-1]['bbox'][1]:regions[n-1]['bbox'][3]] = 0
print(len(n_shapes))
cent =regions[area[-1][1]-1]['centroid']  #centroid of largest area?
print(cent)
plt.imshow(n_shapes,cmap=plt.cm.gray)
plt.plot(cent[1],cent[0],'ro',linewidth=10)
plt.show()


# In[35]:

res = np.zeros_like(liver)
outline = (1-canny(n_shapes,.5))   
n = 1
shape1 = morphology.disk(n)
outline = morphology.erosion(outline,shape1).astype(np.uint8)


for i in range(3):
    res[:,:,i]=liver[:,:,i]*outline




# res = np.zeros_like(liver)
# for i in range(3):
#     res[:,:,i]=liver[:,:,i]*(1-canny(n_shapes))
    
    
plt.imshow(1-n_shapes)
plt.axis("off")
plt.savefig("LARGE_area.png",dpi=800)
plt.show()


# In[36]:



plt.imshow(res)
plt.tight_layout()
plt.savefig('res.png',dpi=800)
plt.show()

