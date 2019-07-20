// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

// NeighInhib adds an additional inhibition factor based on the same
// feature along an orthogonal angle -- assumes inner-most X axis
// represents angle of gabor or related feature.
type NeighInhib struct {
	On bool    `desc:"use neighkborhood inhibition"`
	Gi float32 `def:"0.6" desc:"overall value of the inhibition -- this is what is added into the unit Gi inhibition level"`
}

var (
	// ortho neighbor coordinates for 4 angles, also uses negated version
	// -- = X=0,Y=1
	// /  = X=-1,Y=1
	// |  = X=1,Y=0
	// \  = X=-1,Y=-1
	Neigh4X = []int{0, -1, 1, -1}
	Neigh4Y = []int{1, 1, 0, -1}
)

func (ni *NeighInhib) Defaults() {
	ni.Gi = 0.6
}
