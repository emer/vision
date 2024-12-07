// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
)

// NeighInhib adds an additional inhibition factor based on the same
// feature along an orthogonal angle -- assumes inner-most X axis
// represents angle of gabor or related feature.
// This helps reduce redundancy of feature code.
type NeighInhib struct {

	// use neighborhood inhibition
	On bool

	// overall value of the inhibition -- this is what is added into the unit Gi inhibition level
	Gi float32 `default:"0.6"`
}

var (
	// ortho neighbor coordinates for 4 angles, also uses negated version
	//  .
	// --- = (0,1) (X,Y)
	// . /
	//  /  = (-1,1)
	// | .  = (1,0)
	//  \
	// . \  = (-1,-1)
	Neigh4X = []int{0, -1, 1, -1}
	Neigh4Y = []int{1, 1, 0, -1}
)

func (ni *NeighInhib) Defaults() {
	ni.On = true
	ni.Gi = 0.6
}

// Inhib4 computes the neighbor inhibition on activations
// into extGi.  If extGi is not same shape as act, it will be
// made so (most efficient to re-use same structure).
// Act must be a 4D tensor with features as inner 2D.
// 4 version ONLY works with 4 angles (inner-most feature dimension)
func (ni *NeighInhib) Inhib4(act, extGi *tensor.Float32) {
	extGi.SetShapeSizes(act.Shape().Sizes...)
	gis := extGi.Values

	layY := act.DimSize(0)
	layX := act.DimSize(1)

	plY := act.DimSize(2)
	plX := act.DimSize(3)
	plN := plY * plX

	pi := 0
	for ly := 0; ly < layY; ly++ {
		for lx := 0; lx < layX; lx++ {
			pui := pi * plN
			ui := 0
			for py := 0; py < plY; py++ {
				for ang := 0; ang < plX; ang++ {
					idx := pui + ui
					gi := float32(0)
					npX := lx + Neigh4X[ang]
					npY := ly + Neigh4Y[ang]
					if npX >= 0 && npX < layX && npY >= 0 && npY < layY {
						gi = math32.Max(gi, ni.Gi*act.Value(npY, npX, py, ang))
					}
					nnX := lx - Neigh4X[ang]
					nnY := ly - Neigh4Y[ang]
					if nnX >= 0 && nnX < layX && nnY >= 0 && nnY < layY {
						gi = math32.Max(gi, ni.Gi*act.Value(nnY, nnX, py, ang))
					}
					gis[idx] = gi
					ui++
				}
			}
			pi++
		}
	}
}
