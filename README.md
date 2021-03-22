# NiftiGenerator
NiftiGenerator is a tool to ingest Nifti images using Nibabel, apply basic augmentation, and utilize them as inputs to a deep learning model

Data is sampled as fixed-size chunks. The chunks can be as small as you would like or as large as an entire image.
Augmentation is currently only 2D (not performed on the 3rd dimension of the input images in consideration of non-isotropic input data).
Sampling of each chunk of data is performed after any augmentation.

please see the source code for implementation details. Basic implementations are as follows:

## SingleNiftiGenerator -- To use for generating a single input into your model, do something like the following:
```
    # define the NiftiGenerator
    niftiGen = NiftiGenerator.SingleNiftiGenerator()
    # get augmentation options (see help for get_default_augOptions for more details! )
    niftiGen_augment_opts = NiftiGenerator.SingleNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    # get normalization options ( see help for get_default_normOptions for more details! )
    niftiGen_norm_opts = NiftiGenerator.SingleNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    # initialize the generator (where x_data_train is either a path to a single folder or a list of Nifti files)
    niftiGenTrain.initialize( x_data_train, augOptions=niftiGen_augment_opts, normOptions=niftiGen_norm_opts )
    ## in your training function you will then call something like:
    NiftiGenerator.generate_chunks( niftiGen, chunk_size=(128,128,5), batch_size=16 ) 
    ## to generate a batch of 16, 128x128x5 chunks of data
```

## PairedNiftiGenerator -- To use for generating paired inputs into your model, do something like the following:
```
    # define the NiftiGenerator
    niftiGen = NiftiGenerator.PairedNiftiGenerator()
    # get augmentation options (see help for get_default_augOptions for more details! )
    niftiGen_augment_opts = NiftiGenerator.PairedNiftiGenerator.get_default_augOptions()
    niftiGen_augment_opts.hflips = True
    niftiGen_augment_opts.vflips = True
    niftiGen_augment_opts.rotations = 5
    niftiGen_augment_opts.scalings = .1
    # get normalization options ( see help for get_default_normOptions for more details! )
    niftiGen_norm_opts = NiftiGenerator.PairedNiftiGenerator.get_default_normOptions()
    niftiGen_norm_opts.normXtype = 'auto'
    niftiGen_norm_opts.normYtype = 'fixed'
    niftiGen_norm_opts.normYoffset = 0
    niftiGen_norm_opts.normYscale = 50000
    # initialize the generator (where x_data_train and y_data_train are either a paths to a single folder or lists of Nifti files)
    niftiGen.initialize( x_data_train, y_data_train, augOptions=niftiGen_augment_opts, normOptions=niftiGen_norm_opts )
    ## in your training function you will then call something like:
    NiftiGenerator.generate_chunks( niftiGen, chunk_size=(32,32,32), batch_size=64 ) 
    ## to generate a batch of 64, 32x32x32 chunks of paired data
```

## More advanced things:

    The NiftiGenerators are designed to allow flexible callbacks at various places to do more advanced things to the input data.
    Custom functions can be used in three different places:
    
        1. During augmentation using the augOptions.additionalFunction. This additional function will be called at the last step of the augmentation.
        2. During normalization using the normType ='function'. This function will be called to do the normalization of the input data.
                Note that this is slow because it requires loading the whole Nifti volume
        3. During the sampling of each batch by passing the additional function batchTransformFunction to the initialize call.
                This function will be called right before each batch is returned. The transform function should preserve the size of the batch. 

## Alternatives:

There are a number of other alternatives that you should consider:
    
* TorchIO - https://github.com/fepegar/torchio
* nifti_image_generator - https://github.com/sremedios/nifti_image_generator
    
