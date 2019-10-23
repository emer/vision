// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"image/color"

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
	bd := img.Bounds()
	sz := bd.Size()
	tsr.SetShape([]int{sz.Y + 2*padWidth, sz.X + 2*padWidth}, nil, []string{"Y", "X"})
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := img.At(bd.Min.X+x, bd.Min.Y+sy)
			var cl gi.Color
			cl.SetColor(cv)
			r, g, b, _ := cl.ToFloat32()
			gv := (r + g + b) / 3
			tsr.Set([]int{y + padWidth, x + padWidth}, gv)
		}
	}
}

// GreyTensorToImage converts a greyscale tensor to image -- uses
// existing img if it is of correct size, otherwise makes a new one.
// padWidth is the amount of padding to subtract from all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func GreyTensorToImage(img *image.Gray, tsr *etensor.Float32, padWidth int, topZero bool) *image.Gray {
	var sz image.Point
	sz.Y = tsr.Dim(0) - 2*padWidth
	sz.X = tsr.Dim(1) - 2*padWidth
	if img == nil {
		img = image.NewGray(image.Rectangle{Max: sz})
	} else {
		isz := img.Bounds().Size()
		if isz != sz {
			img = image.NewGray(image.Rectangle{Max: sz})
		}
	}
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := tsr.Value([]int{y + padWidth, x + padWidth})
			iv := uint8(cv * 255)
			img.Set(x, sy, color.Gray{iv})
		}
	}
	return img
}

// WrapPad wraps given padding width of float32 image around sides -- i.e., padding for
// left side of image is the (mirrored) bits from the right side of image, etc.
func WrapPad(tsr *etensor.Float32, padWidth int) {
	sz := image.Point{tsr.Dim(1), tsr.Dim(0)}
	usz := sz
	usz.Y -= padWidth
	usz.X -= padWidth
	for y := 0; y < sz.Y; y++ {
		sy := y
		if y < padWidth {
			sy = usz.Y - (padWidth - y)
		} else if y >= usz.Y {
			sy = padWidth + (y - usz.Y)
		}
		for x := 0; x < padWidth; x++ {
			wv := tsr.Value([]int{sy, usz.X - (padWidth - x)})
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
			sx = usz.X - (padWidth - x)
		} else if x >= usz.X {
			sx = padWidth + (x - usz.X)
		}
		for y := 0; y < padWidth; y++ {
			wv := tsr.Value([]int{usz.Y - (padWidth - y), sx})
			tsr.Set([]int{y, x}, wv)
		}
		for y := usz.Y; y < sz.Y; y++ {
			wv := tsr.Value([]int{padWidth + (y - usz.Y), sx})
			tsr.Set([]int{y, x}, wv)
		}
	}
}
