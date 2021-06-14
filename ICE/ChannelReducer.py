# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper for using sklearn.decomposition on high-dimensional tensors.
Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

import numpy as np
import sklearn.decomposition
import sklearn.cluster


from sklearn.base import BaseEstimator 

ALGORITHM_NAMES = {}
for name in dir(sklearn.decomposition):
    obj = sklearn.decomposition.__getattribute__(name)
    if isinstance(obj, type) and issubclass(obj, BaseEstimator):
        ALGORITHM_NAMES[name] = 'decomposition'
for name in dir(sklearn.cluster):
    obj = sklearn.cluster.__getattribute__(name)
    if isinstance(obj, type) and issubclass(obj, BaseEstimator):
        ALGORITHM_NAMES[name] = 'cluster'


class ChannelDecompositionReducer(object):

    def __init__(self, n_components=3, reduction_alg="NMF", **kwargs):

        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)

        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.decomposition):
            obj = sklearn.decomposition.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError("Unknown dimensionality reduction method '%s'." % reduction_alg)


        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)
        self._is_fit = False

    def _apply_flat(cls, f, acts):
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, acts.shape[-1]])
        new_flat = f(acts_flat)
        if not isinstance(new_flat, np.ndarray):
            return new_flat
        shape = list(orig_shape[:-1]) + [-1]
        return new_flat.reshape(shape)

    def fit(self, acts):
        if hasattr(self._reducer,'partial_fit'):
            res = self._apply_flat(self._reducer.partial_fit, acts)
        else:
            res = self._apply_flat(self._reducer.fit, acts)
        self._is_fit = True
        return res

    def fit_transform(self, acts):
        res = self._apply_flat(self._reducer.fit_transform, acts)
        self._is_fit = True
        return res

    def transform(self, acts):
        res = self._apply_flat(self._reducer.transform, acts)
        return res

    def inverse_transform(self, acts):
        if hasattr(self._reducer,'inverse_transform'):
            res = self._apply_flat(self._reducer.inverse_transform, acts)
        else:
            res = np.dot(acts,self._reducer.components_)
        return res


class ChannelClusterReducer(object):

    def __init__(self, n_components=3, reduction_alg="KMeans", **kwargs):


        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)

        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.cluster):
            obj = sklearn.cluster.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError("Unknown dimensionality reduction method '%s'." % reduction_alg)


        self.n_components = n_components
        self._reducer = reduction_alg(n_clusters=n_components, **kwargs)
        self._is_fit = False

    def _apply_flat(self, f, acts):
        """Utility for applying f to inner dimension of acts.
        Flattens acts into a 2D tensor, applies f, then unflattens so that all
        dimesnions except innermost are unchanged.
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, acts.shape[-1]])
        new_flat = f(acts_flat)
        if not isinstance(new_flat, np.ndarray):
            return new_flat
        shape = list(orig_shape[:-1]) + [-1]
        new_flat = new_flat.reshape(shape)


        if new_flat.shape[-1] == 1:
            new_flat = new_flat.reshape(-1)
            t_flat = np.zeros([new_flat.shape[0],self.n_components])
            t_flat[np.arange(new_flat.shape[0]),new_flat] = 1
            new_flat = t_flat.reshape(shape)

        return new_flat

    def fit(self, acts):
        if hasattr(self._reducer,'partial_fit'):
            res = self._apply_flat(self._reducer.partial_fit, acts)
        else:
            res = self._apply_flat(self._reducer.fit, acts)
        self._reducer.components_ = self._reducer.cluster_centers_
        self._is_fit = True
        return res

    def fit_predict(self, acts):
        res = self._apply_flat(self._reducer.fit_predict, acts)
        self._reducer.components_ = self._reducer.cluster_centers_
        self._is_fit = True
        return res

    def transform(self, acts):
        res = self._apply_flat(self._reducer.predict, acts)
        return res

    def inverse_transform(self, acts):
        res = np.dot(acts,self._reducer.components_)
        return res
