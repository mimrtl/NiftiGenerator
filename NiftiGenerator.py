import nibabel as nib
import cv2
from glob import glob
import types
import os
import sys
import numpy as np
import logging
from scipy.ndimage import affine_transform

module_logger = logging.getLogger(__name__)
module_logger_handler = logging.StreamHandler()
module_logger_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
module_logger_handler.setFormatter(module_logger_formatter)
module_logger.addHandler(module_logger_handler)
module_logger.setLevel(logging.INFO)

# data generator for a single set of nifti files
class SingleNiftiGenerator:
    inputFilesX = []
    augOptions = types.SimpleNamespace()
    normOptions = types.SimpleNamespace()
    normXready = []
    normXoffset = []
    normXscale = []

    def initialize(self, inputX, augOptions=None, normOptions=None):

        # if input is a list, let's just use that
        # otherwise consider this input as a folder
        if isinstance( inputX, list ):
            self.inputFilesX = inputX
        else:
            self.inputFilesX = glob( os.path.join(inputX,'*.nii.gz'),recursive=True) + glob( os.path.join(inputX,'*.nii'),recursive=True)
        num_Xfiles = len(self.inputFilesX)

        module_logger.info( '{} datasets were found'.format(num_Xfiles) )

        if augOptions is None:
            module_logger.warning( 'No augmentation options were specified.' )
            self.augOptions = SingleNiftiGenerator.get_default_augOptions()
        else:
            self.augOptions = augOptions

        if normOptions is None:
            module_logger.warning( 'No normalization options were specified.' )
            self.normOptions = SingleNiftiGenerator.get_default_normOptions()
        else:
            self.normOptions = normOptions

        # handle normalization
        if self.normOptions.normXtype == 'auto'.lower():
            self.normXready = [False] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        elif self.normOptions.normXtype == 'fixed'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [self.normOptions.normXoffset] * num_Xfiles
            self.normXscale = [self.normOptions.normXscale] * num_Xfiles
        elif self.normOptions.normXtype == 'function'.lower():
            self.normXready = [False] * num_Xfiles
        elif self.normOptions.normXtype == 'none'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        else:
            module_logger.error('Fatal Error: Normalization for X was specified as an unknown value.')
            sys.exit(1)

    def generate(self, img_size=(256,256), slice_samples=1, batch_size=16):

        while True:
            # create empty variables for this batch
            batch_X = np.zeros( [batch_size,img_size[0],img_size[1],slice_samples] )

            for i in range(batch_size):
                # get a random subject
                j = np.random.randint( 0, len(self.inputFilesX) )
                currImgFileX = self.inputFilesX[j]

                # load nifti header
                module_logger.debug( 'reading file {}'.format(currImgFileX) )
                Ximg = nib.load( currImgFileX )

                XimgShape = Ximg.header.get_data_shape()

                # determine sampling range
                if slice_samples==1:
                    z = np.random.randint( 0, XimgShape[2]-1 )
                elif slice_samples==3:
                    z = np.random.randint( 1, XimgShape[2]-2 )
                elif slice_samples==5:
                    z = np.random.randint( 2, XimgShape[2]-3 )
                elif slice_samples==7:
                    z = np.random.randint( 3, XimgShape[2]-4 )
                elif slice_samples==9:
                    z = np.random.randint( 4, XimgShape[2]-5 )
                else:
                    module_logger.error('Fatal Error: Number of slice samples must be 1, 3, 5, 7, or 9')
                    sys.exit(1)

                module_logger.debug( 'sampling range is {}'.format(z) )                

                 # handle input data normalization and sampling
                if self.normOptions.normXtype == 'function'.lower():
                    # normalization is performed via a specified function
                    # get normalized data (and read whole volume)
                    tmpX = self.normOptions.normXfunction( Ximg.get_fdata() )
                    # sample data
                    XimgSlices = tmpX[:,:,z-slice_samples//2:z+slice_samples//2+1]
                else:
                    # type is none, auto, or fixed
                    # prepare normalization
                    if not self.normXready[j]:
                        tmpX = Ximg.get_fdata()
                        self.normXoffset[j] = np.mean( tmpX )
                        self.normXscale[j] = np.std( tmpX )
                        self.normXready[j] = True
                    # sample data
                    XimgSlices = Ximg.slicer[:,:,z-slice_samples//2:z+slice_samples//2+1].get_fdata()
                    # do normalization
                    XimgSlices = (XimgSlices - self.normXoffset[j]) / self.normXscale[j]

                # resize to fixed size for model (note img is resized with CUBIC)
                XimgSlices = cv2.resize( XimgSlices, dsize=(img_size[1],img_size[0]), interpolation = self.normOptions.normXinterp )

                # ensure 3D matrix if batch size is equal to 1
                if XimgSlices.ndim == 2:
                    XimgSlices = XimgSlices[...,np.newaxis]

                # augmentation here
                M = self.get_augment_transform()
                XimgSlices = self.do_augment( XimgSlices, M )

                # if an additional augmentation function is supplied, apply it here
                if self.augOptions.additionalFunction:
                    XimgSlices = self.augOptions.additionalFunction( XimgSlices )

                # put into data array for batch for this batch of samples
                batch_X[i,:,:,:] = XimgSlices

                yield( batch_X )

    def get_default_normOptions():
        normOptions = types.SimpleNamespace()
        # set normalization options
        #  type can be 'none', 'auto', 'fixed', 'function'
        # for none, no normalization is done
        # for auto, a Z-score normalization is done on the Nifti volume to make mean=0, stdev=1
        # for fixed, a specified offset and scaling factor is applied (data-offset)/scale
        # for function, a python function is passed that takes the input data and returns a normalized version        
        normOptions.normXtype = 'none'
        normOptions.normXoffset = 0
        normOptions.normXscale = 1
        normOptions.normXfunction = None
        # interp can be any of the opencv interpolation types: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
        # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.LANCZOS4,
        # cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX
        normOptions.normXinterp = cv2.INTER_CUBIC

        return normOptions

    def get_default_augOptions():
        augOptions = types.SimpleNamespace()
        # augmode
        ## choices=['mirror','nearest','reflect','wrap']
        ## help='Determines how the augmented data is extended beyond its boundaries. See scipy.ndimage documentation for more information'
        augOptions.augmode = 'reflect'
        # augseed
        ## help='Random seed (as integer) to set for reproducible augmentation'
        augOptions.augseed = 813
        # addnoise
        ## help='Add Gaussian noise by this (floating point) factor'
        augOptions.addnoise = 0
        # hflips
        ## help='Perform random horizontal flips'
        augOptions.hflips = False
        # vflips
        ## help='Perform random horizontal flips'
        augOptions.vflips = False
        # rotations
        ## help='Perform random rotations up to this angle (in degrees)'
        augOptions.rotations = 0
        # scalings
        ## help='Perform random scalings between the range [(1-scale),(1+scale)]')
        augOptions.scalings = 0
        # shears
        ## help='Add random shears by up to this angle (in degrees)'
        augOptions.shears = 0
        # translations
        ## help='Perform random translations by up to this number of pixels'
        augOptions.translations = 0
        # additional post-processing as function (run after augmentation)
        augOptions.additionalFunction = None

        return augOptions

    def get_augment_transform( self ):
        # use affine transformations as augmentation
        M = np.eye(3)
        # horizontal flips
        if self.augOptions.hflips:
            M_ = np.eye(3)
            M_[1][1] = 1 if np.random.random()<0.5 else -1
            M = np.matmul(M,M_)
        # vertical flips
        if self.augOptions.vflips:
            M_ = np.eye(3)
            M_[0][0] = 1 if np.random.random()<0.5 else -1
            M = np.matmul(M,M_)
        # rotations
        if np.abs( self.augOptions.rotations ) > 1e-2:
            rot_angle = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][0] = np.cos(rot_angle)
            M_[0][1] = np.sin(rot_angle)
            M_[1][0] = -np.sin(rot_angle)
            M_[1][1] = np.cos(rot_angle)
            M = np.matmul(M,M_)
        # shears
        if np.abs( self.augOptions.shears ) > 1e-2:
            rot_angle_x = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            rot_angle_y = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][1] = np.tan(rot_angle_x)
            M_[1][0] = np.tan(rot_angle_y)
            M = np.matmul(M,M_)
        # scaling (also apply specified resizing [--imsize] here)
        if np.abs( self.augOptions.scalings ) > 1e-4:
            init_factor_x = 1
            init_factor_y = 1
            if np.abs( self.augOptions.scalings ) > 1e-4:
                random_factor_x = np.random.randint(-np.abs(self.augOptions.scalings)*10000,np.abs(self.augOptions.scalings)*10000)/10000
                random_factor_y = np.random.randint(-np.abs(self.augOptions.scalings)*10000,np.abs(self.augOptions.scalings)*10000)/10000
            else:
                random_factor_x = 0
                random_factor_y = 0
            scale_factor_x = init_factor_x + random_factor_x
            scale_factor_y = init_factor_y + random_factor_y
            M_ = np.eye(3)
            M_[0][0] = scale_factor_x
            M_[1][1] = scale_factor_y
            M = np.matmul(M,M_)
        # translations
        if np.abs( self.augOptions.translations ) > 0:
            translate_x = np.random.randint( -np.abs( self.augOptions.translations ), np.abs( self.augOptions.translations ) )
            translate_y = np.random.randint( -np.abs( self.augOptions.translations ), np.abs( self.augOptions.translations ) )
            M_ = np.eye(3)
            M_[0][2] = translate_x
            M_[1][2] = translate_y
            M = np.matmul(M,M_)

        return M

    def do_augment( self, X, M ):
        # now apply the transform
        X_ = np.zeros_like(X)

        for k in range(X.shape[2]):
            X_[:,:,k] = affine_transform( X[:,:,k], M, output_shape=X[:,:,k].shape, mode=self.augOptions.augmode )

        # optionally add noise
        if np.abs( self.augOptions.addnoise ) > 1e-10:
            noise_mean = 0
            noise_sigma = self.augOptions.addnoise
            noise = np.random.normal( noise_mean, noise_sigma, X_[:,:,2].shape ) # [:,:,k] for k=0,1,2. Which k? output_shape was undefined 3rd arg here
            for k in range(X_.shape[2]):
                X_[:,:,k] = X_[:,:,k] + noise

        return X_

# data generator for a paired set of nifti files
class PairedNiftiGenerator(SingleNiftiGenerator):
    inputFilesY = []
    normYready = []
    normYoffset = []
    normYscale = []

    def initialize(self, inputX, inputY, augOptions=None, normOptions=None):

        # if input is a list, let's just use that
        # otherwise consider this input as a folder
        if isinstance( inputX, list ):
            self.inputFilesX = inputX
        else:
            self.inputFilesX = sorted( glob( os.path.join(inputX,'*.nii.gz'),recursive=True) + glob( os.path.join(inputX,'*.nii'),recursive=True) )

        if isinstance( inputY, list ):
            self.inputFilesY = inputY
        else:
            self.inputFilesY = sorted( glob( os.path.join(inputY,'*.nii.gz'),recursive=True) + glob( os.path.join(inputY,'*.nii'),recursive=True) )

        num_Xfiles = len(self.inputFilesX)
        num_Yfiles = len(self.inputFilesY)
        module_logger.info( '{} datasets were found for X'.format(num_Xfiles) )
        module_logger.info( '{} datasets were found for Y'.format(num_Yfiles) )

        if num_Xfiles != num_Yfiles:
            module_logger.error( 'Fatal Error: Mismatch in number of datasets.' )
            sys.exit(1)

        if augOptions is None:
            module_logger.warning( 'No augmentation options were specified.' )
            self.augOptions = PairedNiftiGenerator.get_default_augOptions()
        else:
            self.augOptions = augOptions

        if normOptions is None:
            module_logger.warning( 'No normalization options were specified.' )
            self.normOptions = PairedNiftiGenerator.get_default_normOptions()
        else:
            self.normOptions = normOptions

        # handle normalization
        if self.normOptions.normXtype == 'auto'.lower():
            self.normXready = [False] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        elif self.normOptions.normXtype == 'fixed'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [self.normOptions.normXoffset] * num_Xfiles
            self.normXscale = [self.normOptions.normXscale] * num_Xfiles
        elif self.normOptions.normXtype == 'function'.lower():
            self.normXready = [False] * num_Xfiles
        elif self.normOptions.normXtype == 'none'.lower():
            self.normXready = [True] * num_Xfiles
            self.normXoffset = [0] * num_Xfiles
            self.normXscale = [1] * num_Xfiles
        else:
            module_logger.error('Fatal Error: Normalization for X was specified as an unknown value.')
            sys.exit(1)

        if self.normOptions.normYtype == 'auto'.lower():
            self.normYready = [False] * num_Yfiles
            self.normYoffset = [0] * num_Yfiles
            self.normYscale = [1] * num_Yfiles
        elif self.normOptions.normYtype == 'fixed'.lower():
            self.normYready = [True] * num_Yfiles
            self.normYoffset = [self.normOptions.normYoffset] * num_Yfiles
            self.normYscale = [self.normOptions.normYscale] * num_Yfiles
        elif self.normOptions.normYtype == 'function'.lower():
            self.norYready = [False] * num_Yfiles
        elif self.normOptions.normYtype == 'none'.lower():
            self.normYready = [True] * num_Yfiles
            self.normYoffset = [0] * num_Yfiles
            self.normYscale = [1] * num_Yfiles
        else:
            module_logger.error('Fatal Error: Normalization for Y was specified as an unknown value.')
            sys.exit(1)

    def get_default_normOptions():
        normOptions = types.SimpleNamespace()

        # set normalization options
        #  type can be 'none', 'auto', 'fixed', 'function'
        # for none, no normalization is done
        # for auto, a Z-score normalization is done on the Nifti volume to make mean=0, stdev=1
        # for fixed, a specified offset and scaling factor is applied (data-offset)/scale
        # for function, a python function is passed that takes the input data and returns a normalized version
        normOptions.normXtype = 'none'
        normOptions.normXoffset = 0
        normOptions.normXscale = 1
        normOptions.normXfunction = None
        # interp can be any of the opencv interpolation types: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
        # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.LANCZOS4,
        # cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX
        normOptions.normXinterp = cv2.INTER_CUBIC

        normOptions.normYtype = 'none'
        normOptions.normYoffset = 0
        normOptions.normYscale = 1
        normOptions.normYinterp = cv2.INTER_CUBIC
        normOptions.normXfunction = None

        return normOptions

    def generate(self, img_size=(256,256), slice_samples=1, batch_size=16):

        while True:
            # create empty variables for this batch
            batch_X = np.zeros( [batch_size,img_size[0],img_size[1],slice_samples] )
            batch_Y = np.zeros( [batch_size,img_size[0],img_size[1],slice_samples] )

            for i in range(batch_size):
                # get a random subject
                j = np.random.randint( 0, len(self.inputFilesX) )
                currImgFileX = self.inputFilesX[j]
                currImgFileY = self.inputFilesY[j]

                # load nifti header
                module_logger.debug( 'reading files {}, {}'.format(currImgFileX,currImgFileY) )
                Ximg = nib.load( currImgFileX )
                Yimg = nib.load( currImgFileY )

                XimgShape = Ximg.header.get_data_shape()
                YimgShape = Yimg.header.get_data_shape()

                if not XimgShape == YimgShape:
                    module_logger.warning('input data ({} and {}) is not the same size. this may lead to unexpected results or errors!'.format(currImgFileX,currImgFileY))

                # determine sampling range
                if slice_samples==1:
                    z = np.random.randint( 0, XimgShape[2]-1 )
                elif slice_samples==3:
                    z = np.random.randint( 1, XimgShape[2]-2 )
                elif slice_samples==5:
                    z = np.random.randint( 2, XimgShape[2]-3 )
                elif slice_samples==7:
                    z = np.random.randint( 3, XimgShape[2]-4 )
                elif slice_samples==9:
                    z = np.random.randint( 4, XimgShape[2]-5 )
                else:
                    module_logger.error('Fatal Error: Number of slice samples must be 1, 3, 5, 7, or 9')
                    sys.exit(1)                    

                module_logger.debug( 'sampling range is {}'.format(z) )

                 # handle input data normalization and sampling
                if self.normOptions.normXtype == 'function'.lower():
                    # normalization is performed via a specified function
                    # get normalized data (and read whole volume)
                    tmpX = self.normOptions.normXfunction( Ximg.get_fdata() )
                    # sample data
                    XimgSlices = tmpX[:,:,z-slice_samples//2:z+slice_samples//2+1]
                else:
                    # type is none, auto, or fixed
                    # prepare normalization
                    if not self.normXready[j]:
                        tmpX = Ximg.get_fdata()
                        self.normXoffset[j] = np.mean( tmpX )
                        self.normXscale[j] = np.std( tmpX )
                        self.normXready[j] = True
                    # sample data
                    XimgSlices = Ximg.slicer[:,:,z-slice_samples//2:z+slice_samples//2+1].get_fdata()
                    # do normalization
                    XimgSlices = (XimgSlices - self.normXoffset[j]) / self.normXscale[j]

                if self.normOptions.normYtype == 'function'.lower():
                    # normalization is performed via a specified function
                    # get normalized data (and read whole volume)
                    tmpY = self.normOptions.normYfunction( Yimg.get_fdata() )
                    # sample data
                    YimgSlices = tmpY[:,:,z-slice_samples//2:z+slice_samples//2+1]
                else:
                    # type is none, auto, or fixed
                    # prepare normalization                    
                    if not self.normYready[j]:
                        tmpY = Yimg.get_fdata()
                        self.normYoffset[j] = np.mean( tmpY )
                        self.normYscale[j] = np.std( tmpY )
                        self.normYready[j] = True
                    # sample data
                    YimgSlices = Yimg.slicer[:,:,z-slice_samples//2:z+slice_samples//2+1].get_fdata()
                    # do normalization
                    YimgSlices = (YimgSlices - self.normYoffset[j]) / self.normYscale[j]

                # resize to fixed size for model (note img is resized with CUBIC)
                XimgSlices = cv2.resize( XimgSlices, dsize=(img_size[1],img_size[0]), interpolation = self.normOptions.normXinterp)
                YimgSlices = cv2.resize( YimgSlices, dsize=(img_size[1],img_size[0]), interpolation = self.normOptions.normYinterp)

                # ensure 3D matrix if batch size is equal to 1
                if XimgSlices.ndim == 2:
                    XimgSlices = XimgSlices[...,np.newaxis]
                if YimgSlices.ndim == 2:
                    YimgSlices = YimgSlices[...,np.newaxis]

                # augmentation here
                M = self.get_augment_transform()
                XimgSlices = self.do_augment( XimgSlices, M )
                YimgSlices = self.do_augment( YimgSlices, M )

                # if an additional augmentation function is supplied, apply it here
                if self.augOptions.additionalFunction:
                    XimgSlices = self.augOptions.additionalFunction( XimgSlices )
                    YimgSlices = self.augOptions.additionalFunction( YimgSlices )

                # put into data array for batch for this batch of samples
                batch_X[i,:,:,:] = YimgSlices
                batch_Y[i,:,:,:] = XimgSlices

                yield (batch_X , batch_Y)
