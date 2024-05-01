# vision

This repository contains visual processing packages in Go (golang), focused mainly on providing efficient V1 (primary visual cortex) level filtering of images, with the output then suitable as input for neural networks.

Two main types of filters are supported:

* **Gabor** filters simulate V1 simple-cell responses in terms of an oriented sine wave times a gaussian envelope that localizes the filter in space.  This produces an edge detector that detects oriented contrast transitions between light and dark.  In general, the main principle of primary visual filtering is to focus on spatial (and temporal) changes, while filtering out static, uniform areas.

* **DoG** (difference of gaussian) filters simulate retinal On-center vs. Off-center contrast coding cells -- unlike gabor filters, these do not have orientation tuning.  Mathematically, they are a difference between a narrow (center) vs wide (surround) gaussian, of opposite signs, balanced so that a uniform input generates offsetting values that sum to zero.  In the visual system, orientation tuning is constructed from aligned DoG-like inputs, but it is more efficient to just use the Gabor filters directly.  However, DoG filters capture the "blob" cells that encode color contrasts.

The `vfilter` package contains general-purpose filtering code that applies (convolves) any given filter with a visual input.  It also supports converting an `image.Image` into a `tensor.Float32` tensor which is the main data type used in this framework.  It also supports max-pooling for efficiently reducing the dimensionality of inputs.

The `kwta` package provides an implementation of the feedforward and feedback (FFFB) inhibition dynamics (and noisy X-over-X-plus-1 activation function) from the `Leabra` algorithm to produce a k-Winners-Take-All processing of visual filter outputs -- this increases the contrast and simplifies the representations, and is a good model of the dynamics in primary visual cortex.


