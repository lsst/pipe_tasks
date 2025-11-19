# This file is part of trailNet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["TrailMLTask", "TrailMLConfig", "TrailMLConnections", "NNModelPackage", "TrailMLInterface"]

import lsst.geom
import lsst.pex.config
import lsst.pipe.base
from lsst.utils.timer import timeMethod
import numpy as np
import astropy.table as at

import torch
import os
import importlib.util

class TrailMLConnections(lsst.pipe.base.PipelineTaskConnections,
                             dimensions=("instrument", "visit", "detector")):
    
    science = lsst.pipe.base.connectionTypes.Input(
        doc="Input post-isr image.",
        dimensions=("instrument", "exposure", "detector"),
        storageClass="Exposure",
        name="post_isr_image"
    )

    # uncomment this when using model from butler memory

    # model = lsst.pipe.base.connectionTypes.Input(
    #     doc="Input model.",
    #     dimensions=("instrument", ),
    #     storageClass="NNModelPackagePayload",
    #     name="trailmotion_model"
    # )

    classifications = lsst.pipe.base.connectionTypes.Output(
        doc="Catalog of trailing classification for each visit, "
            "per detector image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ArrowAstropy",
        name="trail_labels_detector",
    )

class TrailMLConfig(lsst.pipe.base.PipelineTaskConfig, pipelineConnections=TrailMLConnections):
    # comment this out when using model from butler memory
    modelArchPath = lsst.pex.config.Field(
        optional=True,
        dtype=str,
        doc=("Filesystem path to the model architecture Python file (e.g., trail_model.py).")
    )
    modelWeightsPath = lsst.pex.config.Field(
        optional=True,
        dtype=str,
        doc=("Filesystem path to the pretrained model weights checkpoint.")
    )

class TrailMLTask(lsst.pipe.base.PipelineTask):
    """Task for running TrailML classification on the post-isr images.
    """
    _DefaultName = "trailML"
    ConfigClass = TrailMLConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.butler_loaded_package = None
        # comment this put when using model from butler memory
        self.interface = TrailMLInterface(self.config.modelArchPath, self.config.modelWeightsPath)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        # inputs['idGenerator'] = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        dataid = butlerQC.quantum.dataId
        print("dataid : ", dataid)
        outputs = self.run(inputs['science'], dataid['visit'], dataid['detector'])

        # uncomment this when using model from butler memory
        # outputs = self.run(inputs['science'], dataid['visit'], dataid['detector'], inputs['model'])

        butlerQC.put(outputs, outputRefs)


    @timeMethod
    # def run(self, science, visit, detector, model):
    def run(self, science, visit, detector):
        # self.butler_loaded_package = pretrainedModel  # This will be used by the interface

        # uncomment the following lines when using model from butler memory
        # self.interface = TrailMLInterface()
        # torch.load the model from memory
        

        # takes in one image i.e. one CCD at a time
        labels, probs = self.interface.infer(science)
        print('visit: ', visit, 'detector: ', detector, 'probs: ', probs, 'labels: ', labels)

        # write into an AstropyQTable
        
        classifications = at.QTable()
        classifications["visit"] = [visit]
        classifications["detector"] = [detector]
        classifications["probs"] = [probs]
        classifications["labels"] = [labels]

        return lsst.pipe.base.Struct(classifications=classifications)


# class to load the neural network model
class NNModelPackage:
    """
    An interface to load physical storage of network architecture &
    pretrained models out of clients' code.
    """

    def __init__(self, model_arch_path, model_weights_path, **kwargs):
        self.model_arch_path = model_arch_path
        self.model_weights_path = model_weights_path

    def load_arch(self, device):
        """
        Load and return the model architecture from file storage
        (no loading of pre-trained weights).

        Parameters
        ----------
        device : `torch.device`
            Device to load the model on.

        Returns
        -------
        model : `torch.nn.Module`
            The model architecture. The exact type of this object
            is model-specific.

        See Also
        --------
        load_weights
        """
        # Expect a filesystem path to the model architecture python file
        # (e.g., .../trail_model.py) to be provided on self.model_arch_path.
        model_path = getattr(self, "model_arch_path", None)
        if not model_path:
            raise ValueError(
                "NNModelPackage.model_arch_path is not set; expected a path to the trail_model script."
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model architecture file not found: {model_path}")

        # Dynamically import the module from the provided path
        spec = importlib.util.spec_from_file_location("trail_model", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load spec for module at path: {model_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Retrieve the expected architecture class from the module
        if not hasattr(module, "TrailClassifierNet"):
            raise ImportError(
                f"TrailClassifierNet not found in module loaded from: {model_path}"
            )

        ModelClass = getattr(module, "TrailClassifierNet")
        model = ModelClass(in_channels=1)
        return model.to(device)

    def load_weights(self, device):
        """
        Load and return a checkpoint of a neural network model.

        Parameters
        ----------
        device : `torch.device`
            Device to load the pretrained weights on.

        Returns
        -------
        network_data : `dict`
            Dictionary containing a saved network state in PyTorch format,
            composed of the trained weights, optimizer state, and other
            useful metadata.

        See Also
        --------
        load_arch
        """

        network_data = torch.load(self.model_weights_path, map_location=device,
                                  weights_only=True)
        return network_data


    def load(self, device):
        """Load model architecture and pretrained weights.
        This method handles all different modes of storages.


        Parameters
        ----------
        device : `str`
            Device to create the model on, e.g. 'cpu' or 'cuda:0'.

        Returns
        -------
        model : `torch.nn.Module`
            The neural network model, loaded with pretrained weights.
            Its type should be a subclass of nn.Module, defined by
            the architecture module.
        """

        # Check if the specified device is valid.
        if device not in ['cpu'] + ['cuda:%d' % i for i in range(torch.cuda.device_count())]:
            raise ValueError("Invalid device: %s" % device)

        # Load various components.
        # Note that because of the way the StorageAdapterButler works,
        # the model architecture and the pretrained weights are loaded
        # into the cpu memory, and only then moved to the target device.
        model = self.load_arch(device='cpu')
        network_data = self.load_weights(device='cpu')

        # Load pretrained weights into model
        model.load_state_dict(network_data)

        # Move model to the specified device, if it is not already there.
        if device != 'cpu':
            model = model.to(device)

        return model


class TrailMLInterface:
    """Interface for TrailMLTask to interact with TrailML models.
    """

    def __init__(self, model_arch_path, model_weights_path):
        self.model_package = NNModelPackage(model_arch_path, model_weights_path)
        self.init_model()

    def init_model(self):
        """Initialize the TrailML model.
        """
        # Load the model on CPU for initialization
        self.model = self.model_package.load(device='cpu')
        self.model.eval()

    @staticmethod
    def apply_mean_kernel(image_array: np.ndarray, kernel_size: int) -> np.ndarray:
        """mean over kernel_size squares."""
        reduced_x = image_array.shape[0] // kernel_size
        reduced_y = image_array.shape[1] // kernel_size
        image_array_reduced = np.zeros((reduced_x, reduced_y), dtype=float)
        for j in range(reduced_x):
            for k in range(reduced_y):
                image_array_reduced[j, k] = np.mean(
                    image_array[kernel_size*j:kernel_size*(j+1),
                                kernel_size*k:kernel_size*(k+1)]
                )
        return image_array_reduced
    
    @staticmethod
    def truncation_transform(image: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """sigma-clipped stats and thresholding to 0/1."""
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(image)
        return np.where(image > median + threshold * std, 1, 0)

    def prepare_input(self, image):
        """Preprocess the input image for inference.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input post-isr image array.   

        Returns
        -------
        input_tensor : `torch.Tensor`
            The preprocessed input tensor for the model.
        """
        # check the image is a numpy array
        image = image.image.array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")
        image = self.apply_mean_kernel(image, 16)
        if image.shape == (250, 256):
            image = image[:, 1:-1]

        # apply truncation transform
        image = self.truncation_transform(image, threshold=10)
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.transpose((0, 3, 1, 2))).float()

        return image


    def infer(self, image, label_threshold=0.5, device='cpu'):
        """Run inference on the input image using the TrailML model.
        This inference method takes in only one image at a time. 
        This can be updated to take a lot of images at a time, split them into batches,
        and run inference on batches for better performance.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image data for inference.
        label_threshold : `float`, optional
            The threshold to classify the output scores (default is 0.5).
        device : `str`, optional
            The device to run inference on (default is 'cpu').

        Returns
        -------
        scores : `list`
            List of classification scores for the input image.
        """

        # Load the model
        model = self.model_package.load(device=device)
        model.eval()

        # Preprocess the image 
        input_image = self.prepare_input(image)


        # Run inference
        with torch.no_grad():
            logits = model(input_image.to(device))
            y_pred = torch.sigmoid(logits).squeeze(1)
            label = (y_pred.cpu().numpy() >= label_threshold).astype(int)[0]

        # show only 3 decimal places
        prob = y_pred.cpu().numpy()[0]
        prob = np.round(prob, 3)

        return label, prob


    