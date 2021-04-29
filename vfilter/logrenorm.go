// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"math"

	"github.com/emer/etable/etensor"
	"github.com/emer/etable/norm"
	"github.com/goki/mat32"
)

// TensorLogNorm32 computes 1 + log of all the numbers and then does
// Max Div renorm so result is normalized in 0-1 range.
// computed on the first ndim dims of the tensor, where 0 = all values,
// 1 = norm each of the sub-dimensions under the first outer-most dimension etc.
// ndim must be < NumDims() if not 0 (panics).
func TensorLogNorm32(tsr *etensor.Float32, ndim int) {
	for i, v := range tsr.Values {
		tsr.Values[i] = mat32.Log(1 + v)
	}
	norm.TensorDivNorm32(tsr, ndim, norm.Max32)
}

// TensorLogNorm64 computes 1 + log of all the numbers and then does
// Max Div renorm so result is normalized in 0-1 range.
// computed on the first ndim dims of the tensor, where 0 = all values,
// 1 = norm each of the sub-dimensions under the first outer-most dimension etc.
// ndim must be < NumDims() if not 0 (panics).
func TensorLogNorm64(tsr *etensor.Float64, ndim int) {
	for i, v := range tsr.Values {
		tsr.Values[i] = math.Log(1 + v)
	}
	norm.TensorDivNorm64(tsr, ndim, norm.Max64)
}
