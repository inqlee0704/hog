import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_differential_filter():
    filter_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return filter_x, filter_y

def filter_image(im, filter):
    im_filtered = np.zeros((np.size(im,0),np.size(im,1)))
    im_pad = np.pad(im,((1,1),(1,1)), 'constant')

    tracker_i=-1
    for i in range(np.size(im_filtered,0)):
        tracker_i+=1
        tracker_j=-1
        for j in range(np.size(im_filtered,1)):
            tracker_j+=1
            v=0
            for k in range(np.size(filter,0)):
                for l in range(np.size(filter,1)):
                    v += filter[k][l] * im_pad[k+tracker_i][l+tracker_j]
            im_filtered[i,j] = v
    return im_filtered

def get_gradient(im_dx, im_dy):
    grad_mag = (im_dx**2 + im_dy**2)**0.5
    grad_angle = np.arctan2(im_dy,im_dx)
    for i in range(np.size(grad_angle,0)):
        for j in range(np.size(grad_angle,1)):
            if grad_angle[i,j] < 0:
                grad_angle[i,j] += np.pi
    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    c = cell_size
    m = np.size(grad_mag,0)
    n = np.size(grad_mag,1)
    M = m//c
    N = n//c

    ori_histo = np.zeros((M,N,6))
    grad_angle = grad_angle*(180/np.pi)

    for i in range(M):
        for j in range(N):
            for u in range(c):
                for v in range(c):
                    u1 = u + i*c
                    v1 = v + j*c
                    if(165<=grad_angle[u1,v1]<=180) or (0<=grad_angle[u1,v1]<15):
                        ori_histo[i,j,0] += grad_mag[u1,v1]
                    elif 15<=grad_angle[u1,v1]<45:
                        ori_histo[i,j,1] += grad_mag[u1,v1]
                    elif 45<=grad_angle[u1,v1]<75:
                        ori_histo[i,j,2] += grad_mag[u1,v1]
                    elif 75<=grad_angle[u1,v1]<105:
                        ori_histo[i,j,3] += grad_mag[u1,v1]
                    elif 105<=grad_angle[u1,v1]<135:
                        ori_histo[i,j,4] += grad_mag[u1,v1]
                    elif 135<=grad_angle[u1,v1]<165:
                        ori_histo[i,j,5] += grad_mag[u1,v1]

    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    M = np.size(ori_histo,0)
    N = np.size(ori_histo,1)

    ori_histo_normalized = np.zeros((M-(block_size-1),\
                                    N-(block_size-1),\
                                    6*block_size*block_size))

    e=0.001
    for i in range(M-(block_size-1)):
        for j in range(N-(block_size-1)):
            base = e**2
            t1 = ori_histo[i,j,:]
            t2 = ori_histo[i,j+1,:]
            t3 = ori_histo[i+1,j,:]
            t4 = ori_histo[i+1,j+1,:]
            ori_histo_normalized[i][j][:] = np.concatenate((t1,t2,t3,t4),axis=0)
            for k in range(6*block_size**block_size):
                base += ori_histo_normalized[i][j][k]**2
            base = base**0.5
            ori_histo_normalized[i][j][:] = ori_histo_normalized[i][j][:]/base

    return ori_histo_normalized

def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    cell_size = 8
    block_size = 2

    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im,filter_x)
    im_dy = filter_image(im,filter_y)

    grad_mag, grad_angle = get_gradient(im_dx,im_dy)
    ori_histo = build_histogram(grad_mag,grad_angle,cell_size)
    ori_histo_normalized = get_block_descriptor(ori_histo,block_size)
    im = im*255
    visualize_hog_cell(im,ori_histo,cell_size)
    x = np.size(ori_histo_normalized,0)
    y = np.size(ori_histo_normalized,1)
    z = np.size(ori_histo_normalized,2)
    hog = np.reshape(ori_histo_normalized,(x*y*z,1))
    return hog

def visualize_hog_cell(im, ori_histo, cell_size):
    norm_constant = 1e-3
    num_cell_h, num_cell_w, num_bins = ori_histo.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2: cell_size*num_cell_w: cell_size], np.r_[cell_size/2: cell_size*num_cell_h: cell_size])
    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2) # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
        color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.title("HOG_cell")
    plt.show()


def visualize_hog_block(im, hog, cell_size, block_size):
# visualize histogram of each block
    num_bins = 6
    max_len = 7 # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[int(cell_size*block_size/2): cell_size*num_cell_w-(cell_size*block_size/2)+1: cell_size], np.r_[int(cell_size*block_size/2): cell_size*num_cell_h-(cell_size*block_size/2)+1: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
        color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.title("HOG_block")
    plt.show()

def visualize_hog(im, ori_histo, cell_size):
    norm_constant = 1e-3
    num_bins = ori_histo.shape[2]
    height, width = im.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2:width:cell_size], np.r_[cell_size/2:height:cell_size])

    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2)  # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len  # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
        color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

if __name__=='__main__':
    # img_path = 'cameraman.jpg'
    im = cv2.imread(img_path, 0)
    im = im.astype('float') /255.0

    #Visualization
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im,filter_x)
    im_dy = filter_image(im,filter_y)
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    f.add_subplot(1,3,2)
    plt.imshow(im_dx, cmap='gray', vmin=np.min(im_dx), vmax=np.max(im_dx))
    plt.title('im_dx')
    f.add_subplot(1,3,3)
    plt.imshow(im_dy, cmap='gray', vmin=np.min(im_dy), vmax=np.max(im_dy))
    plt.title('im_dy')
    plt.show()

    hog = extract_hog(im)
    cell_size = 8
    block_size =2
    visualize_hog_block(im, hog, cell_size, block_size)
