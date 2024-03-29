// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import "image"

// Geom contains the filtering geometry info for a given filter pass.
type Geom struct {

	// size of input -- computed from image or set
	In image.Point

	// size of output -- computed
	Out image.Point

	// starting border into image -- must be >= FiltRt
	Border image.Point

	// spacing -- number of pixels to skip in each direction
	Spacing image.Point

	// full size of filter
	FiltSz image.Point

	// computed size of left/top size of filter
	FiltLt image.Point

	// computed size of right/bottom size of filter (FiltSz - FiltLeft)
	FiltRt image.Point
}

// Set sets the basic geometry params
func (ge *Geom) Set(border, spacing, filtSz image.Point) {
	ge.Border = border
	ge.Spacing = spacing
	ge.FiltSz = filtSz
	ge.UpdtFilt()
}

// LeftHalf returns the left / top half of a filter
func LeftHalf(x int) int {
	if x%2 == 0 {
		return x / 2
	}
	return (x - 1) / 2
}

// UpdtFilt updates filter sizes, and ensures that Border >= FiltRt
func (ge *Geom) UpdtFilt() {
	ge.FiltLt.X = LeftHalf(ge.FiltSz.X)
	ge.FiltLt.Y = LeftHalf(ge.FiltSz.Y)
	ge.FiltRt = ge.FiltSz.Sub(ge.FiltLt)
	if ge.Border.X < ge.FiltRt.X {
		ge.Border.X = ge.FiltRt.X
	}
	if ge.Border.Y < ge.FiltRt.Y {
		ge.Border.Y = ge.FiltRt.Y
	}
}

// SetSize sets the input size, and computes output from that.
func (ge *Geom) SetSize(inSize image.Point) {
	ge.In = inSize
	b2 := ge.Border.Mul(2)
	av := ge.In.Sub(b2)
	ge.Out = av.Div(ge.Spacing.X) // only 1
}
