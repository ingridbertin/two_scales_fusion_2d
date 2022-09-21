import numpy as np
import skimage
from PIL import Image

class ContrastStretching():
    def __init__(self, image, name):
         self.image = image
         self.name = name

    def take_view(self, x, indices, axis=0):
        '''
        Like `np.take` but returns a view instead of a copy.
        '''
        make_indices = lambda a: slice(None) if a != axis else indices
        idx = tuple(make_indices(ax) for ax in range(x.ndim))
        return x[idx]


    def quantile_transform(self, x, quantiles=(0.02, 0.98), n_samples=1000000, clip=True, channel_axis=0, out=None):
        '''
        Scale data to the [0, 1] interval in terms of quantiles. Clips outliers if specified.
        Make `out=x` for an inplace transform.
        '''
        x_out = out if out is not None else np.empty_like(x)
        n_channels = x.shape[channel_axis]
        
        # table: voxel x channel value
        table = np.moveaxis(x, channel_axis, -1).reshape(-1, n_channels)
        idx_sample = np.random.choice(len(table), n_samples)
        table_sample = table[idx_sample]

        for c in range(n_channels):
            # get sample quantiles
            channel_sample = table_sample[..., c]
            vmin, vmax = np.quantile(channel_sample, quantiles)
            
            # transform entire array
            channel = self.take_view(x, c, axis=channel_axis)
            out_channel = self.take_view(x_out, c, axis=channel_axis)
            
            out_channel[...] = np.round((channel - vmin) / float(vmax - vmin), 1)
            if clip:
                out_channel[...] = np.clip(out_channel, 0, 1)
        
        return x_out

    def add_channels(self):
        """
        Adds channels to grayscale image
        
        Returns
        -------
        float_image_3_channels: array_like(float, ndim=3)
            image with 3 channels needed for using the quantile_transform function
        
        """    
        img =  Image.open(self.image)
        float_image = np.array(img, dtype=np.float32)
        # add a channel axis
        float_image_with_axis = np.expand_dims(float_image, -1)
        # repeat the data over this axis
        float_image_3_channels = float_image_with_axis.repeat(3, axis=-1)
        return float_image_3_channels

    def save_image(self, array_image):
        """
        Saves the image obtained after transforming the histogram in the range [0,1]
        
        Parameters
        ----------
        array_image: array_like(float, ndim=3)
            This is an array with the image values obtained after transformation
        """    
        # save image in tif
        skimage.io.imsave(fname= './imgs/' + self.name + '.tif', arr=array_image[:, :, 0])
        return
    
if __name__ == "__main__":
    new_image = ContrastStretching()
    new_image_with_3_channels = new_image.add_channels()
    new_image_percentile_stretching = new_image.quantile_transform(new_image_with_3_channels)
    new_image.save_image(new_image_percentile_stretching)