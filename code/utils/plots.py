

def get_plot_planes(mask, split):
    
    if split == "hemi":
        mask = mask[:mask.shape[0]//2, :, :]
    
    props = measure.regionprops(mask)
    planes = props[0].centroid
    
    return [int(p) for p in planes]

def plot_overlay(img, mask, plane):
    
    mask = np.where(mask == 0, np.nan, mask)
    vmax = mask.max()
    
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
    
    
    i = img[plane[0], :, :]
    m = mask[plane[0], :, :]
    ax[0].imshow(i)
    ax[0].imshow(m, cmap = 'jet_r', vmax = vmax, alpha = 0.6)
    
    i = img[:, plane[1], :]
    m = mask[:, plane[1], :]
    ax[1].imshow(i)
    ax[1].imshow(m, cmap = 'jet_r', vmax = vmax, alpha = 0.6)
    
    i = img[:, :, plane[2]]
    m = mask[:, :, plane[2]]
    ax[2].imshow(i)
    ax[2].imshow(m, cmap = 'jet_r', vmax = vmax, alpha = 0.6)

def plot_images(template, img, plane = 200):
     
    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    
    i = img[plane, :, :]
    t = template[plane, :, :]
    ax[0][0].imshow(t)
    ax[1][0].imshow(i)
    
    i = img[:, plane, :]
    t = template[:, plane, :]
    ax[0][1].imshow(t)
    ax[1][1].imshow(i)
    
    i = img[:, :, plane]
    t = template[:, :, plane]
    ax[0][2].imshow(t)
    ax[1][2].imshow(i)

def plot_cells_only(cells):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(cells[:, 0], cells[:, 1], cells[:, 2])

def plot_cells(template, cells, plane = 200, vmax = 450):
    
    if isinstance(plane, int):
        plane = [plane] * 3
    
    cells = np.array(cells, dtype = int)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
    
    c = cells[cells[:, 0] == plane[0], :]
    t = template[plane[0], :, :]
    ax[0].imshow(t, vmax = vmax)
    ax[0].scatter(c[:, 2], c[:, 1])
    
    
    c = cells[cells[:, 1] == plane[1], :]
    t = template[:, plane[1], :]
    ax[1].imshow(t, vmax = vmax)
    ax[1].scatter(c[:, 2], c[:, 0])
    
    
    c = cells[cells[:, 2] == plane[2], :]
    t = template[:, :, plane[2]]
    ax[2].imshow(t, vmax = vmax)
    ax[2].scatter(c[:, 1], c[:, 0])

def plot_warps(ccf, template):
    
    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    
    ax[0][0].imshow(ccf[:, :, 200, 0])
    ax[0][1].imshow(ccf[:, 200, :, 0])
    ax[0][2].imshow(ccf[200, :, :, 0])
    
    ax[1][0].imshow(template[:, :, 200, 0])
    ax[1][1].imshow(template[:, 200, :, 0])
    ax[1][2].imshow(template[200, :, :, 0])