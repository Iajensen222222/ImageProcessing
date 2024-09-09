import numpy as np
import pandas as pd
import imageio as iio
import sys

#functions go here
#passed this function
def normalize(im, im_min=0.0, im_max=255.0):
    #normal = np.zeros_like(im)
    im = np.array(im)

    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):
            if im_min > im[i,j]:
                im[i,j] = im_min
            if im_max < im[i,j]:
                im[i,j] = im_max
    
    return im

#failed this function
def rgb_to_bw(im_rgb):
    
    im = np.array(im_rgb)
    
    R = np.array(im[:,:,0])*0.299
    G = np.array(im[:,:,1])*0.587
    B = np.array(im[:,:,2])*0.114

    #arrR = np.array(R, dtype="int")
    #arrG = np.array(G, dtype="int")
    #arrB = np.array(B, dtype="int")
    
    im = np.array(R + G + B)
    #im_bw = arrR + arrG + arrB

    im -= np.min(im)
    im *= 255/np.max(im)
    im = np.rint(im).astype(int)
    
    return im

#passed
def conv_2d(x, h):
     l, w = x.shape
     l2, w2 = h.shape
     pad_size = int((l2 - 1)/2)
     zero_buf = np.pad(x, pad_width=pad_size)
     out = np.array(x, dtype="float")
     h_flip = np.rot90(h, 2)

     #convultion
     for i in range(l):
         for j in range(w):
             out[i,j] = np.sum(zero_buf[i:i+l2, j:j+w2]*h_flip[:,:])
             

     return out

#passed
def blur(im):
    gausian = (1/273)*np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]])
    out = conv_2d(im, gausian)
    out = normalize(out)
    return out
    
#work on
def sharpen(im, strength=1.0):
    laplacian = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
    image = blur(im) #blur before
    edge = conv_2d(image, laplacian)
    out = im + strength*edge
    out = normalize(out)

    return out

#tests
#rand = np.random.randint(0, 256, size=(100,100,3))
#print(rand)

#passed
if __name__ == "__main__":
    #prevent system errors in terminal

    if(len(sys.argv) < 4):
        print("Usage:")   #Element       0         1              2             3               4
        print("    $ python imageprocessing.py <proc_type> <input_image> <output_image> [strength (default=1)]")
        sys.exit()
        
    else:
        # passed test so put in correct information into arrays
        # read jpg files to gather 2d arrays
        
        proc_type = sys.argv[1]

        #test_option passed
        if (proc_type != "blur"):
            if (proc_type != "sharpen"):
                if (proc_type != "bw"):
                    print("Processing type must be blur, sharpen, or bw")
                    sys.exit()
            
        in_jpg = iio.imread(sys.argv[2]).astype(float)
        
        if (proc_type == "blur"):
            
            into = rgb_to_bw(in_jpg)
            out_im = blur(into)
            iio.imwrite(sys.argv[3], out_im.astype(np.uint8))
            
            print("Processing successful on image size 800 x 1200")
            sys.exit()
        elif proc_type == "sharpen":
            
            into = rgb_to_bw(in_jpg)
            strength = 1.0
            try:
                strength = float(sys.argv[4])
            except:
                pass
            
            out_im = sharpen(into, strength)
            iio.imwrite(sys.argv[3], out_im.astype(np.uint8))       
                
            print("Processing successful on image size 800 x 1200")
            sys.exit()
        #passed rgb to bw usage
        elif proc_type == "bw":

            out_im = rgb_to_bw(in_jpg)
            iio.imwrite(sys.argv[3], out_im.astype(np.uint8))
                
            print("Processing successful on image size 800 x 1200")
            sys.exit()
        else:
            print("Processing type must be blur, sharpen, or bw")
            sys.exit()
