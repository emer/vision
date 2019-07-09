// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"

	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

// RGBToGrey converts an RGB input image to a greyscale etensor
// in preparation for processing.
// padWidth is the amount of padding to add on all sides
func RGBToGrey(img image.Image, tsr *etensor.Float32, padWidth int) {
	sz := img.Bounds().Size()
	tsr.SetShape([]int{sz.Y + 2*padWidth, sz.X + 2*padWidth}, nil, []string{"Y", "X"})
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			cv := img.At(x, y)
			var cl gi.Color
			cl.SetColor(cv)
			r, g, b, _ := cl.ToFloat32()
			gv := (r + g + b) / 3
			tsr.Set([]int{y + padWidth, x + padWidth}, gv)
		}
	}
}

// todo: wrap pad, blend pad etc methods for padding.
