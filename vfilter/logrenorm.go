// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"math"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/norm"
	"cogentcore.org/core/tensor/stats/stats"
)

// TensorLogNorm computes 1 + log of all the numbers and then does
// Max Div renorm so result is normalized in 0-1 range.
// computed on the first ndim dims of the tensor, where 0 = all values,
// 1 = norm each of the sub-dimensions under the first outer-most dimension etc.
// ndim must be < NumDims() if not 0 (panics).
func TensorLogNorm(tsr tensor.Tensor, ndim int) {
	switch tt := tsr.(type) {
	case *tensor.Float32:
		for i, v := range tt.Values {
			tt.Values[i] = math32.Log(1 + v)
		}
	case *tensor.Float64:
		for i, v := range tt.Values {
			tt.Values[i] = math.Log(1 + v)
		}
	default:
		norm.FloatOnlyError()
	}

	norm.TensorDivNorm(tsr, ndim, stats.Max32, stats.Max64)
}
