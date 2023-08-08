// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

import (
	"github.com/emer/etable/etensor"
	"github.com/goki/mat32"
)

// NeighInhib adds an additional inhibition factor based on the same
// feature along an orthogonal angle -- assumes inner-most X axis
// represents angle of gabor or related feature.
// This helps reduce redundancy of feature code.
type NeighInhib struct {

	// use neighborhood inhibition
	On bool `desc:"use neighborhood inhibition"`

	// [def: 0.6] overall value of the inhibition -- this is what is added into the unit Gi inhibition level
	Gi float32 `def:"0.6" desc:"overall value of the inhibition -- this is what is added into the unit Gi inhibition level"`
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
func (ni *NeighInhib) Inhib4(act, extGi *etensor.Float32) {
	if !extGi.Shape.IsEqual(&act.Shape) {
		extGi.SetShape(act.Shape.Shp, act.Shape.Strd, act.Shape.Nms)
	}
	gis := extGi.Values

	layY := act.Dim(0)
	layX := act.Dim(1)

	plY := act.Dim(2)
	plX := act.Dim(3)
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
						gi = mat32.Max(gi, ni.Gi*act.Value([]int{npY, npX, py, ang}))
					}
					nnX := lx - Neigh4X[ang]
					nnY := ly - Neigh4Y[ang]
					if nnX >= 0 && nnX < layX && nnY >= 0 && nnY < layY {
						gi = mat32.Max(gi, ni.Gi*act.Value([]int{nnY, nnX, py, ang}))
					}
					gis[idx] = gi
					ui++
				}
			}
			pi++
		}
	}
}
