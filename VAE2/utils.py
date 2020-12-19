from torch.nn import init
from matplotlib import pyplot as plt

def display_images(in_, out, n=1):
    '''
    Output input and output images. It help visualizing the
    input image and the output image of an autoencoder

    Parameters:
        in_ (torch.Tensor): Input tensor converting to images
        out (torch.Tensor): Output tensor converting to images
        n (int): Number of lines
    '''
    for N in range(n):
        if in_ is not None:
            in_pic = in_.data.cpu().view(-1, 256, 256)
            plt.figure(figsize=(18, 4))
            plt.suptitle('Real test data / reconstructions', color='w', fontsize=16)
            for i in range(4):
                plt.subplot(1,4,i+1)
                plt.imshow(in_pic[i+4*N])
                plt.axis('off')
        out_pic = out.data.cpu().view(-1, 256, 256)
        plt.figure(figsize = (18, 6))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(out_pic[i+4*N])
            plt.axis('off')
