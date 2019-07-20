// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package vfilter provides filtering methods for the vision package.
These apply etensor.Tensor filters to a 2D visual input via Conv
(convolution) function, using filter-parallel approach:
Each go routine does a different filter in a set of filters,
e.g., different angles of Gabor filters.  This is coarse-grained,
strictly parallel, and thus very efficient.

image.go contains routines for converting an image into the float32
etensor.Float32 that is required for doing the convolution.
* RGBToGrey converts an RGB image to a greyscale float32.

MaxPool function does Max-pooling over filtered results to reduce
dimensionality, consistent with standard DCNN approaches.

Geom manages the geometry for going from an input image to the
filtered output of that image.

Unlike the C++ version, no wrapping or clipping is supported directly:
all input images must be padded so that the filters can be applied with
appropriate padding border, guaranteeing that there are no bounds issues.
See WrapPad for wrapping-based padding.
*/
package vfilter
