# Config for makePsfMatchingKernel; experimental based on Martin's new Config

from pexConfig import Config, Field
class PsfMatchingConfig(Config):
    warpingConfig = ConfigField(
        afwMath.WarpConfig,
        doc = "Config for warping exposures to a common alignment",
        optional=False,
    )

    detectionConfig = ConfigField(
        ipDiffim.DetectionConfig,
        doc = "Config for detection of objects for psf matching",
        optional=False,
    )

    ######
    #
    # Fitting for background using Afw
    #
    useAfwBackground = Field(
        bool,
        doc = "Use afw background subtraction instead of ip_diffim",
        default = False,
    )

    afwBackgroundConfig = ConfigField(
        ipDiffim.AfwBackgroundConfig,
        doc = """Config in case diffim needs to do Afw background subtraction, e.g. for object detection
                or when fitting for background using Afw exclusively (option useAfwBackground)""",
        default = ipDiffim.AfwBackgroundConfig(),
    )

    ######
    #
    # Do you fit for background *at all* here
    #
    fitForBackground = Field(
        bool,
        doc = "Include terms (including kernel cross terms) for background",
        default = False,
    )

    ######
    #
    # Do we use createDefaultConfig to modify terms based on FWHM?
    #
    scaleByFwhm = Field(
        bool,
        doc = "Scale kernelSize, alardSigGauss, and fpGrowPix by input Fwhm",
        default = False,
    )

    ######
    #
    # Basis set selection
    #
    kernelBasisSet = ChoiceField(
        doc = "Type of basis set for PSF matching kernel.",
        default = "alard-lupton",
        allowed = {
            "alard-lupton": """Alard-Lupton sum-of-gaussians basis set
                     * The first term has no spatial variation
                     * The kernel sum is conserved
                     * Default for usePcaForSpatialKernel is False""",
            "delta-function": """Delta-function kernel basis set
                     * You may enable the option useRegularization
                     * Default for usePcaForSpatialKernel is True, which enables
                       kernel sum conservation for delta-function kernels""",
        },
    )

    ######
    #
    # Kernel size
    #
    kernelSize = Field(
        int,
        doc = """Number of rows/columns in the convolution kernel; odd-valued.
                Modified by kernelSizeFwhmScaling if scaleByFwhm = True""",
        default = 19,
    )

    kernelSizeFwhmScaling = Field(
        float,
        doc = """How much to scale the kernel size based on the Psf Fwhm;
                should be smaller than fpGrowFwhmScaling.  Sets kernelSize.""",
        default = 4.0,
    )

    kernelSizeMin = Field(
        int,
        doc = "Minimum kernel dimensions",
        default = 7,
    )

    kernelSizeMax = Field(
        int,
        doc = "Maximum kernel dimensions",
        default = 31,
    )

    ######
    #
    # Alard-Lupton Basis Parameters
    #
    alardNGauss = Field(
        int,
        doc = "Number of gaussians in alard-lupton basis",
        default = 3,
    )

    alardDegGauss = ListField(
        int,
        doc = "Degree of spatial modification of gaussians in alard-lupton basis",
        maxLength = 5,
        default = (4, 3, 2),
    )

    alardSigGauss = ListField(
        float,
        doc = """Sigma in pixels of gaussians in alard-lupton basis (note: FWHM = 2.35 sigma). 
                Scaled by alardSigFwhmScaling if scaleByFwhm = True""",
        maxLength = 5,
        default = (0.7, 1.5, 3.0),
    )

    alardSigFwhmScaling = ListField(
        float,
        doc = "Scaling of the alard-lupton gaussian sigmas.  Sets alardSigGauss",
        maxLength = 5,
        default = (0.50, 1.00, 2.00),
    )

    ######
    #
    # Delta Function Basis Parameters
    #
    useRegularization = Field(
        bool,
        doc = "Use regularization to smooth the delta function kernels",
        default = True,
    )
    
    regularizationType = ChoiceField(
        doc = "Type of regularization.",
        default = "centralDifference",
        allowed = {
            "centralDifference": "Penalize second derivative using 2-D stencil",
            "forwardDifference": "Penalize first, second, third or combination of derivatives",
        ),
    )

    centralRegularizationStencil = ChoiceField(
        int,
        doc = "Type of stencil to approximate central derivative (for centralDifference only)",
        default = 9,
        allowed = {
            5: "5-point stencil including only adjacent-in-x,y elements",
            9: "9-point stencil including diagonal elements",
        },
    )

    forwardRegularizationOrders = ListField(
        int,
        doc = "Array showing which order derivatives to penalize (for forwardDifference only)",
        maxLength = 3,
        default = (1, 2),
    )

    regularizationBorderPenalty = Field(
        float,
        doc = "Value of the penalty for kernel border pixels",
        default = 3.0,
    )

    regularizationScaling = Field(
        float,
        doc = """Fraction of the default lambda strength (N.R. 18.5.8) to use. 
                somewhere around 1e-4 to 1e-5 seems to work.
                some kernels need high freq power""",
        default = 1e-4,
    )

    lambdaType = Field(
        string,
        doc = "How to choose the value of the regularization strength",
        default = "absolute",
        allowed = Field(
            name = "absolute",
            doc =  "Use lambdaValue as the value of regularization strength",
        }
        allowed = Field(
            name = "relative",
            doc =  "Use lambdaValue as fraction of the default regularization strength (N.R. 18.5.8)",
        }
        allowed = Field(
            name = "minimizeBiasedRisk",
            doc =  "Minimize biased risk estimate",
        }
        allowed = Field(
            name = "minimizeUnbiasedRisk",
            doc =  "Minimize unbiased risk estimate",
        }
    )
    
    lambdaValue = Field(
        float,
        doc = "Value used for absolute or relative determinations of regularization strength",
        default = 0.2,
    )

    lambdaStepType = Field(
        string,
        doc = """If scan through lambda needed (minimizeBiasedRisk, minimizeUnbiasedRisk) use log
                or linear steps""",
        default = "log",
        allowed = Field(
            name = "log",
            doc = "Step in log intervals; e.g. lambdaMin, lambdaMax, lambdaStep = -1.0, 2.0, 0.1",
        }
        allowed = Field(
            name = "linear",
            doc = "Step in linear intervals; e.g. lambdaMin, lambdaMax, lambdaStep = 0.1, 100, 0.1",
        }
    )
    
    lambdaMin = Field(
        float,
        doc = """If scan through lambda needed (minimizeBiasedRisk, minimizeUnbiasedRisk) 
                start at this value.  If lambdaStepType = log:linear, suggest -1:0.1""",
        default = -1.0,
    )
    
    lambdaMax = Field(
        float,
        doc = """If scan through lambda needed (minimizeBiasedRisk, minimizeUnbiasedRisk) 
                stop at this value.  If lambdaStepType = log:linear, suggest 2:100""",
        default = 2.0,
    )
    
    lambdaStep = Field(
        float,
        doc = """If scan through lambda needed (minimizeBiasedRisk, minimizeUnbiasedRisk) 
                step in these increments.  If lambdaStepType = log:linear, suggest 0.1:0.1""",
        default = 0.1,
    )

    ######
    #
    # Spatial modeling
    #
    spatialKernelType = ChoiceField(
        string,
        doc = "Type of spatial function for kernel",
        default = "chebyshev1",
        allowed = Field(
            name = "chebyshev1",
            doc =  "Chebyshev polynomial of the first kind",
        }
        allowed = Field(
            name = "polynomial",
            doc =  "Standard x,y polynomial",
        }
    )

    spatialKernelOrder = Field(
        int,
        doc = "Spatial order of convolution kernel variation",
        default = 1,
    )

    spatialBgType = Field(
        string,
        doc = "Type of spatial function for kernel",
        default = "chebyshev1",
        allowed = {
            "chebyshev1": "Chebyshev polynomial of the first kind",
            "polynomial": "Standard x,y polynomial",
        }
    )

    spatialBgOrder = Field(
        int,
        doc = "Spatial order of differential background variation",
        default = 1,
    )

    sizeCellX = Field(
        int,
        doc = "Size (rows) in pixels of each SpatialCell for spatial modeling",
        default = 256,
    )

    sizeCellY = Field(
        int,
        doc = "Size (columns) in pixels of each SpatialCell for spatial modeling",
        default = 256,
    )

    nStarPerCell = Field(
        int,
        doc = "Number of KernelCandidates in each SpatialCell to use in the spatial fitting",
        default = 1,
    )

    maxSpatialIterations = Field(
        int,
        doc = "Maximum number of iterations for rejecting bad KernelCandidates in spatial fitting",
        default = 3,
    )

    ######
    #
    # Spatial modeling; Pca
    #
    usePcaForSpatialKernel = Field(
        bool,
        doc = """Use Pca to reduce the dimensionality of the kernel basis sets.
                This is particularly useful for delta-function kernels.
                Functionally, after all Cells have their raw kernels determined, we run 
                a Pca on these Kernels, re-fit the Cells using the eigenKernels and then 
                fit those for spatial variation using the same technique as for Alard-Lupton kernels.
                If this option is used, the first term will have no spatial variation and the 
                kernel sum will be conserved.""",
        default = False, # !!! SHOULD DEPEND ON kernelBasisSet
    )

    subtractMeanForPca = Field(
        bool,
        doc = "Subtract off the mean feature before doing the Pca",
        default = True,
    )

    numPrincipalComponents = Field(
        int,
        doc = """Number of principal components to use for Pca basis, including the 
                mean kernel if requested.""",
        default = 50,
    )

    fracEigenVal = Field(
        float,
        doc = """At what fraction of the eigenvalues do you cut off the expansion.
                Warning: not yet implemented""",
        default = 0.99,
    )

    ######
    # 
    # What types of clipping of KernelCandidates to enable
    #
    singleKernelClipping = Field(
        bool,
        doc = "Do sigma clipping on each raw kernel candidate",
        default = True,
    )

    kernelSumClipping = Field(
        bool,
        doc = "Do sigma clipping on the ensemble of kernel sums",
        default = True,
    )

    spatialKernelClipping = Field(
        bool,
        doc = "Do sigma clipping after building the spatial model",
        default = True,
    )

    ######
    # 
    # Clipping of KernelCandidates based on diffim residuals
    #
    candidateResidualMeanMax = Field(
        float,
        doc = """Rejects KernelCandidates yielding bad difference image quality.
                Represents average over pixels of (image/sqrt(variance)).""",
        default = 0.25,
    )

    candidateResidualStdMax = Field(
        float,
        doc = """Rejects KernelCandidates yielding bad difference image quality.
                Represents stddev over pixels of (image/sqrt(variance)).""",
        default = 1.50,
    )

    useCoreStats = Field(
        bool,
        doc = "Use the core of the stamp for the quality statistics, instead of the entire footprint",
        default = True,
    )

    candidateCoreRadius = Field(
        int,
        doc = """Radius for calculation of stats in 'core' of KernelCandidate diffim.
                Total number of pixels used will be (2*radius)**2. 
                This is used both for 'core' diffim quality as well as ranking of
                KernelCandidates by their total flux in this core""",
        default = 3,
    )

    ######
    # 
    # Clipping of KernelCandidates based on kernel sum distribution
    #
    maxKsumSigma = Field(
        float,
        doc = """Maximum allowed sigma for outliers from kernel sum distribution.
                Used to reject variable objects from the kernel model""",
        default = 3.0,
    )

    ######
    # 
    # Clipping of KernelCandidates based on their matrices
    #
    checkConditionNumber = Field(
        bool,
        doc = """Test for maximum condition number when inverting a kernel matrix?
                Anything above the value is not used and the candidate is set as BAD.        
                Also used to truncate inverse matrix in estimateBiasedRisk.  However,
                if you are doing any deconvolution you will want to turn this off, or use
                a large maxConditionNumber""",
        default = False,
    )

    maxConditionNumber = Field(
        float,
        doc = """Maximum condition number for a well conditioned matrix.
                Suggested values:
                * 5.0e6 for 'delta-function' basis
                * 5.0e7 for 'alard-lupton' basis""",
        default = 5.0e7,
    )

    conditionNumberType = ChoiceField(
        string,
        doc = "Use singular values (SVD) or eigen values (EIGENVALUE) to determine condition number",
        allowed = {
            "SVD": "Use singular values",
            "EIGENVALUE": "Use eigen values (faster)",
        }
        default = "EIGENVALUE",
    )
    
    ######
    # 
    # Fitting of single kernel to object pair
    #
    iterateSingleKernel = Field(
        bool,
        doc = """Remake single kernel using better variance estimate after first pass?
                Primarily useful when convolving a single-depth image, otherwise not necessary.""",
        default = False,
    )

    constantVarianceWeighting = Field(
        bool,
        doc = """Use constant variance weighting in single kernel fitting?
                In some cases this is better for bright star residuals.""",
        default = False,
    )

    calculateKernelUncertainty = Field(
        bool,
        doc = """Calculate kernel and background uncertainties for each kernel candidate?
                This comes from the inverse of the covariance matrix.
                Warning: regularization can cause problems for this step.""",
        default = False,
    )

    ######
    # 
    # Any modifications by subpolicies
    #
    modifiedForImagePsfMatch = Field(
        bool,
        doc = "Config modified for ImagePsfMatch class",
        default = False,
    )

    modifiedForDeconvolution = Field(
        bool,
        doc = "Config modified for deconvolution",
        default = False,
    )

    useOuterForDeconv = Field(
        bool,
        doc = "Use outer fifth gaussian",
        default = False,
    )

    modifiedForModelPsfMatch = Field(
        bool,
        doc = "Config modified for ModelPsfMatch class",
        default = False,
    )

    modifiedForSnapSubtraction = Field(
        bool,
        doc = "Config modified for subtraction of back-to-back snaps",
        default = False,
    )
