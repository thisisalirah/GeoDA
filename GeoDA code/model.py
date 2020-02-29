'''
This file is copied and slightly modified from foolbox.
'''

import numpy as np
import warnings
import torch


class PyTorchModel:
    """Creates a :class:`Model`.

    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    cuda : bool
        A boolean specifying whether the model uses CUDA. If None,
        will default to torch.cuda.is_available()
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """

    def __init__(
            self,
            model,
            bounds,
            num_classes,
            channel_axis=1,
            cuda=None,
            preprocessing=(0, 1)):

        assert len(bounds) == 2
        self._bounds = bounds
        assert channel_axis in [0, 1, 2, 3]
        self._channel_axis = channel_axis
        assert len(preprocessing) == 2
        self._preprocessing = preprocessing

        self._num_classes = num_classes
        self._model = model

        if cuda is None:
            cuda = torch.cuda.is_available()
        self.cuda = cuda

        if model.training:
            warnings.warn(
                'The PyTorch model is in training mode and therefore might'
                ' not be deterministic. Call the eval() method to set it in'
                ' evaluation mode if this is not intended.')

    def bounds(self):
	    return self._bounds

    def _process_input(self, input_):
	    psub, pdiv = self._preprocessing
	    psub = np.asarray(psub, dtype=input_.dtype)
	    pdiv = np.asarray(pdiv, dtype=input_.dtype)
	    result = input_
	    if np.any(psub != 0):
	        result = input_ - psub  # creates a copy
	    if np.any(pdiv != 1):
	        if np.any(psub != 0):  # already copied
	            result /= pdiv  # in-place
	        else:
	            result = result / pdiv  # creates a copy
	    assert result.dtype == input_.dtype
	    return result

    def _process_gradient(self, gradient):
	    _, pdiv = self._preprocessing
	    pdiv = np.asarray(pdiv, dtype=gradient.dtype)
	    if np.any(pdiv != 1):
	        result = gradient / pdiv
	    else:
	        result = gradient
	    assert result.dtype == gradient.dtype
	    return result

    def _old_pytorch(self):
        # lazy import
        import torch
        version = torch.__version__.split('.')[:2]
        pre04 = int(version[0]) == 0 and int(version[1]) < 4
        return pre04

    def batch_predictions(self, images):
        # lazy import
        import torch
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        images = self._process_input(images)
        n = len(images)
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, volatile=True)
            predictions = self._model(images)
            predictions = predictions.data
        else:
            predictions = self._model(images)
            # TODO: add no_grad once we have a solution
            # for models that require grads internally
            # for inference
            # with torch.no_grad():
            #     predictions = self._model(images)
        if self.cuda:  # pragma: no cover
             predictions = predictions.cpu()
        if not self._old_pytorch():
            predictions = predictions.detach()
        predictions = predictions.numpy()
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

    def predictions(self, image):
        return np.squeeze(self.batch_predictions(image[np.newaxis]), axis=0)

    def predictions_and_gradient(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        image = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target)
        if self.cuda:  # pragma: no cover
            target = target.cuda()

        assert image.ndim == 3
        images = image[np.newaxis]
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()

        if self._old_pytorch():  # pragma: no cover
            target = Variable(target)
            images = Variable(images, requires_grad=True)
        else:
            images.requires_grad_()

        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        loss.backward()
        grad = images.grad

        if self._old_pytorch():  # pragma: no cover
            predictions = predictions.data
        if self.cuda:  # pragma: no cover
            predictions = predictions.cpu()

        if not self._old_pytorch():
            predictions = predictions.detach()
        predictions = predictions.numpy()
        predictions = np.squeeze(predictions, axis=0)
        assert predictions.ndim == 1
        assert predictions.shape == (self.num_classes(),)

        if self._old_pytorch():  # pragma: no cover
            grad = grad.data
        if self.cuda:  # pragma: no cover
            grad = grad.cpu()
        if not self._old_pytorch():
            grad = grad.detach()
        grad = grad.numpy()
        grad = self._process_gradient(grad)
        grad = np.squeeze(grad, axis=0)
        assert grad.shape == image.shape

        return predictions, grad

    def _loss_fn(self, image, label):
        # lazy import
        import torch
        import torch.nn as nn
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        image = self._process_input(image)
        target = np.array([label])
        target = torch.from_numpy(target)
        if self.cuda:  # pragma: no cover
            target = target.cuda()
        if self._old_pytorch():  # pragma: no cover
            target = Variable(target)

        images = torch.from_numpy(image[None])
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, volatile=True)
        predictions = self._model(images)
        ce = nn.CrossEntropyLoss()
        loss = ce(predictions, target)
        if self._old_pytorch():  # pragma: no cover
            loss = loss.data
        if self.cuda:  # pragma: no cover
            loss = loss.cpu()
        loss = loss.numpy()
        return loss

    def backward(self, gradient, image):
        # lazy import
        import torch
        if self._old_pytorch():  # pragma: no cover
            from torch.autograd import Variable

        assert gradient.ndim == 1

        gradient = torch.from_numpy(gradient)
        if self.cuda:  # pragma: no cover
            gradient = gradient.cuda()
        if self._old_pytorch():  # pragma: no cover
            gradient = Variable(gradient)

        image = self._process_input(image)
        assert image.ndim == 3
        images = image[np.newaxis]
        images = torch.from_numpy(images)
        if self.cuda:  # pragma: no cover
            images = images.cuda()
        if self._old_pytorch():  # pragma: no cover
            images = Variable(images, requires_grad=True)
        else:
            images.requires_grad_()
        predictions = self._model(images)

        print(predictions.size())
        predictions = predictions[0]

        assert gradient.dim() == 1
        assert predictions.dim() == 1
        assert gradient.size() == predictions.size()

        loss = torch.dot(predictions, gradient)
        loss.backward()
        # should be the same as predictions.backward(gradient=gradient)

        grad = images.grad

        if self._old_pytorch():  # pragma: no cover
            grad = grad.data
        if self.cuda:  # pragma: no cover
            grad = grad.cpu()
        if not self._old_pytorch():
            grad = grad.detach()
        grad = grad.numpy()
        grad = self._process_gradient(grad)
        grad = np.squeeze(grad, axis=0)
        assert grad.shape == image.shape

        return grad
