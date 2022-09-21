import numpy as np
import matplotlib.pyplot as plt
from convert_image_to_array import convertImageToGSLIBFile, convertImageToHardData
import mpslib as mps
from pre_process_data import ContrastStretching


class GenerateImage():
    def __init__(self):
        self.images = ['./imgs/first_scale181a.tif', './imgs/second_scale_1196a.tif']
        # # Choose to compute entropy
        # self.mpslib.par['do_entropy'] = 1
        self.image = []
        self.time = []
    
    def convert_histogram(self):
        for index, i in enumerate(self.images):
            new_image = ContrastStretching(i, str(index + 1) + '_scale')
            new_image_with_3_channels = new_image.add_channels()
            new_image_percentile_stretching = new_image.quantile_transform(new_image_with_3_channels)
            new_image.save_image(new_image_percentile_stretching)

    def create_TI_file(self):
        self.ti = './imgs/2_scale.tif'
        self.dat_ti = convertImageToGSLIBFile(self.ti)
        self.original_ti = mps.eas.read('ti.dat')
        
    def configure_MPS_method(self):
        self.first_scale = './imgs/1_scale.tif'
        # Initialize MPSlib using mps_genesim algorithm, and seetings
        self.mpslib = mps.mpslib(method='mps_genesim')
        self.mpslib.par['simulation_grid_size']=np.array([30*8.92, 30*8.92, 1])
        self.mpslib.par['grid_cell_size']=np.array([3.771*10**(-6),3.771*10**(-6),1])
        self.ncond = np.array([i for i in range(0, 100, 20)])
        self.mpslib.par['n_real'] = 1
        self.mpslib.par['ti_fnam'] = './ti.dat'
        self.mpslib.par['n_threads'] = 4
        self.mpslib.d_hard = convertImageToHardData(self.first_scale)
        return self.mpslib
    
    def saveFigure(self):
        fig1 = plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(np.squeeze(self.original_ti['Dmat'])))
        fig1.savefig('./results_fig/original.png')
        plt.close(fig1)
        for index, ncond in enumerate(self.ncond):
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(np.transpose(np.squeeze(self.image[index])))
            plt.title('CPU time = %.1f' % (self.time[index]) + 's')
            plt.imsave('./results_fig/' + 'Figure_ncond_' + str(ncond) + '.png', np.transpose(np.squeeze(self.image[index])), cmap='gray')
            plt.close(fig)

    def generateFigureTime(self):
        fig2 = plt.figure(figsize=(5, 5))
        plt.plot(self.ncond,self.time,'.')
        plt.grid()
        plt.xlabel('n_cond')
        plt.ylabel('simulation time (s)')
        fig2.savefig('./results_fig/' + 'Figure_ncond_time' + '.png')
        plt.close(fig2)


    def run(self):
        for ncond in self.ncond:
            self.mpslib.par['n_cond'] = ncond
            self.mpslib.run_parallel()
            self.image.append(self.mpslib.sim[-1])
            self.time.append(self.mpslib.time)
        return

if __name__ == "__main__":
    image = GenerateImage()
    image.convert_histogram()
    image.create_TI_file()
    image.configure_MPS_method()
    image.run()
    image.saveFigure()
    image.generateFigureTime()
