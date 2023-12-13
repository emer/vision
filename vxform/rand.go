// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vxform

import (
	"math/rand"

	"goki.dev/etable/v2/minmax"
)

// Rand specifies random transforms
type Rand struct {

	// min -- max range of X-axis (horizontal) translations to generate (as proportion of image size)
	TransX minmax.F32

	// min -- max range of Y-axis (vertical) translations to generate (as proportion of image size)
	TransY minmax.F32

	// min -- max range of scales to generate
	Scale minmax.F32

	// min -- max range of rotations to generate (in degrees)
	Rot minmax.F32
}

// Gen Generates new random transform values
func (rx *Rand) Gen(xf *XForm) {
	trX := rx.TransX.ProjVal(rand.Float32())
	trY := rx.TransY.ProjVal(rand.Float32())
	sc := rx.Scale.ProjVal(rand.Float32())
	rt := rx.Rot.ProjVal(rand.Float32())
	xf.Set(trX, trY, sc, rt)
}
