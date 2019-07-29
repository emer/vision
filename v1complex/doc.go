// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package v1complex implements V1 (primary visual cortex)
complex-cell filters, which operate on the output of
V1 simple-cell (gabor) filters.  Tests indicate that these
filters significantly improve object recognition performance,
beyond what is available from simple-cell output only,
and they are well-established biologically.

Typically the AngleOnly version of V1 simple cell inputs are
used: MaxReduce over the two (On vs. Off) polarities of
gabor filters.

* Length Sum (LenSum) integrates multiple simple-cells along
orientation direction, producing larger-scale features.

* End Stop computes the difference between a length sum input
minus a set of same-orientation stopping features at the end of the
line.  Thus, it responds maximally where a line ends.
The length sum is one to the "left" of the current position
and the off features are one to the "right".
*/
package v1complex
