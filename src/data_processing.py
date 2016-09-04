from abc import ABCMeta, abstractmethod

import numpy as np
import os.path


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


class DataProcessing(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getTrainingSet(*args, **kwargs):
        pass

    @abstractmethod
    def getTestingSet(*args, **kwargs):
        pass

# end class DataProcessing


class Mnist(DataProcessing):

    dirh = os.path.expanduser('~') + '/mnist'

    @staticmethod
    def _build_data_set(imagefile, labelfile, picklename):
        from struct import unpack
        from numpy import zeros, uint8, mat
        import pickle
        import gzip

        """Read input-vector (image) and target class (label, 0-9) and return
           it as list of tuples.
        """

        f = Mnist.dirh + '/' + picklename + '.pkl'
        if os.path.isfile(f):
            with open(f, 'rb') as file:
                data = pickle.load(file)
        else:
            # Open the images with gzip in read binary mode
            images = gzip.open(imagefile, 'rb')
            labels = gzip.open(labelfile, 'rb')

            # Read the binary data
            # We have to get big endian unsigned int. So we need '>I'

            # Get metadata for images
            images.read(4)  # skip the magic_number
            number_of_images = images.read(4)
            number_of_images = unpack('>I', number_of_images)[0]
            rows = images.read(4)
            rows = unpack('>I', rows)[0]
            cols = images.read(4)
            cols = unpack('>I', cols)[0]

            # Get metadata for labels
            labels.read(4)  # skip the magic_number
            N = labels.read(4)
            N = unpack('>I', N)[0]
            print(N)

            if number_of_images != N:
                raise Exception('The number of labels did not match '
                                'the number of images.')

            # Get the data
            x = zeros((rows * cols, N), dtype=uint8)  # Initialize numpy array
            y = zeros((1, N), dtype=uint8)  # Initialize numpy array
            for i in range(N):
                if i % 1000 == 0:
                    print("i: %i" % i)
                for j in range(rows * cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[j][i] = (float(tmp_pixel) / 255)
                tmp_label = labels.read(1)
                y[0][i] = unpack('>B', tmp_label)[0]
            y = Mnist._convert_to_binary(y)
            data = {'x': mat(x), 'y': mat(y), 'rows': rows, 'cols': cols}
            with open(f, 'wb') as file:
                pickle.dump(data, file, -1)
        return data
    # end


    @staticmethod
    def _convert_to_binary(m):
        targets = np.mat(np.zeros((10, m.shape[1]), dtype=np.uint8))
        for i in range(m.shape[1]):
            v = m[0][i]
            targets[v, i] = 1
        return targets
    # end


    @staticmethod
    def getTestingSet(*args, **kwargs):
        return Mnist._build_data_set(
            Mnist.dirh + '/t10k-images-idx3-ubyte.gz',
            Mnist.dirh + '/t10k-labels-idx1-ubyte.gz',
            'mnist_testing'
        )
    # end

    @staticmethod
    def getTrainingSet(*args, **kwargs):
        return Mnist._build_data_set(
            Mnist.dirh + '/train-images-idx3-ubyte.gz',
            Mnist.dirh + '/train-labels-idx1-ubyte.gz',
            'mnist_training'
        )
    #end

# end class Mnist
