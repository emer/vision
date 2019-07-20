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
// padWidth is the amount of padding to add on all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBToGrey(img image.Image, tsr *etensor.Float32, padWidth int, topZero bool) {
	sz := img.Bounds().Size()
	tsr.SetShape([]int{sz.Y + 2*padWidth, sz.X + 2*padWidth}, nil, []string{"Y", "X"})
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := img.At(x, sy)
			var cl gi.Color
			cl.SetColor(cv)
			r, g, b, _ := cl.ToFloat32()
			gv := (r + g + b) / 3
			tsr.Set([]int{y + padWidth, x + padWidth}, gv)
		}
	}
}

// WrapPad wraps given padding width of float32 image around sides
func WrapPad(tsr *etensor.Float32, padWidth int) {
	sz := image.Point{tsr.Dim(1), tsr.Dim(0)}
	usz := sz
	usz.Y -= padWidth
	usz.X -= padWidth
	for y := 0; y < sz.Y; y++ {
		sy := y
		if y < padWidth {
			sy = usz.Y - 1 - y
		} else if y >= usz.Y {
			sy = padWidth + (y - usz.Y)
		}
		for x := 0; x < padWidth; x++ {
			wv := tsr.Value([]int{sy, usz.X - 1 - x})
			// testing only:
			// wv *= float32(x) / float32(padWidth)
			// tsr.Set([]int{sy, usz.X - 1 - x}, wv)
			tsr.Set([]int{y, x}, wv)
		}
		for x := usz.X; x < sz.X; x++ {
			wv := tsr.Value([]int{sy, padWidth + (x - usz.X)})
			tsr.Set([]int{y, x}, wv)
		}
	}
	for x := 0; x < sz.X; x++ {
		sx := x
		if x < padWidth {
			sx = usz.X - 1 - x
		} else if x >= usz.X {
			sx = padWidth + (x - usz.X)
		}
		for y := 0; y < padWidth; y++ {
			wv := tsr.Value([]int{usz.Y - 1 - y, sx})
			tsr.Set([]int{y, x}, wv)
		}
		for y := usz.Y; y < sz.Y; y++ {
			wv := tsr.Value([]int{padWidth + (y - usz.Y)})
			tsr.Set([]int{y, x}, wv)
		}
	}
}
