// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"image/color"

	"cogentcore.org/core/colors"
	"cogentcore.org/core/tensor"
)

// RGBToTensor converts an RGB input image to an RGB tensor
// with outer dimension as RGB components.
// padWidth is the amount of padding to add on all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBToTensor(img image.Image, tsr *tensor.Float32, padWidth int, topZero bool) {
	bd := img.Bounds()
	sz := bd.Size()
	tsr.SetShapeSizes(3, sz.Y+2*padWidth, sz.X+2*padWidth)
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := img.At(bd.Min.X+x, bd.Min.Y+sy)
			r, g, b, _ := colors.ToFloat32(cv)
			tsr.Set(r, 0, y+padWidth, x+padWidth)
			tsr.Set(g, 1, y+padWidth, x+padWidth)
			tsr.Set(b, 2, y+padWidth, x+padWidth)
		}
	}
}

// RGBTensorToImage converts an RGB tensor to image -- uses
// existing image if it is of correct size, otherwise makes a new one.
// tensor must have outer dimension as RGB components.
// padWidth is the amount of padding to subtract from all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBTensorToImage(img *image.RGBA, tsr *tensor.Float32, padWidth int, topZero bool) *image.RGBA {
	var sz image.Point
	sz.Y = tsr.DimSize(1) - 2*padWidth
	sz.X = tsr.DimSize(2) - 2*padWidth
	if img == nil {
		img = image.NewRGBA(image.Rectangle{Max: sz})
	} else {
		isz := img.Bounds().Size()
		if isz != sz {
			img = image.NewRGBA(image.Rectangle{Max: sz})
		}
	}
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			r := tsr.Value(0, y+padWidth, x+padWidth)
			g := tsr.Value(1, y+padWidth, x+padWidth)
			b := tsr.Value(2, y+padWidth, x+padWidth)
			ri := uint8(r * 255)
			gi := uint8(g * 255)
			bi := uint8(b * 255)
			img.Set(x, sy, color.RGBA{ri, gi, bi, 255})
		}
	}
	return img
}

// RGBToGrey converts an RGB input image to a greyscale tensor
// in preparation for processing.
// padWidth is the amount of padding to add on all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBToGrey(img image.Image, tsr *tensor.Float32, padWidth int, topZero bool) {
	bd := img.Bounds()
	sz := bd.Size()
	tsr.SetShapeSizes(sz.Y+2*padWidth, sz.X+2*padWidth)
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := img.At(bd.Min.X+x, bd.Min.Y+sy)
			r, g, b, _ := colors.ToFloat32(cv)
			gv := (r + g + b) / 3
			tsr.Set(gv, y+padWidth, x+padWidth)
		}
	}
}

// GreyTensorToImage converts a greyscale tensor to image -- uses
// existing img if it is of correct size, otherwise makes a new one.
// padWidth is the amount of padding to subtract from all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func GreyTensorToImage(img *image.Gray, tsr *tensor.Float32, padWidth int, topZero bool) *image.Gray {
	var sz image.Point
	sz.Y = tsr.DimSize(0) - 2*padWidth
	sz.X = tsr.DimSize(1) - 2*padWidth
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
			cv := tsr.Value(y+padWidth, x+padWidth)
			iv := uint8(cv * 255)
			img.Set(x, sy, color.Gray{iv})
		}
	}
	return img
}

// WrapPad wraps given padding width of float32 image around sides
// i.e., padding for left side of image is the (mirrored) bits
// from the right side of image, etc.
func WrapPad(tsr *tensor.Float32, padWidth int) {
	sz := image.Point{tsr.DimSize(1), tsr.DimSize(0)}
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
			wv := tsr.Value(sy, usz.X-(padWidth-x))
			tsr.Set(wv, y, x)
		}
		for x := usz.X; x < sz.X; x++ {
			wv := tsr.Value(sy, padWidth+(x-usz.X))
			tsr.Set(wv, y, x)
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
			wv := tsr.Value(usz.Y-(padWidth-y), sx)
			tsr.Set(wv, y, x)
		}
		for y := usz.Y; y < sz.Y; y++ {
			wv := tsr.Value(padWidth+(y-usz.Y), sx)
			tsr.Set(wv, y, x)
		}
	}
}

// WrapPadRGB wraps given padding width of float32 image around sides
// i.e., padding for left side of image is the (mirrored) bits
// from the right side of image, etc.
// RGB version iterates over outer-most dimension of components.
func WrapPadRGB(tsr *tensor.Float32, padWidth int) {
	nc := tsr.DimSize(0)
	for i := 0; i < nc; i++ {
		simg := tsr.SubSpace(i).(*tensor.Float32)
		WrapPad(simg, padWidth)
	}
}

// EdgeAvg returns the average value around the effective edge of image
// at padWidth in from each side
func EdgeAvg(tsr *tensor.Float32, padWidth int) float32 {
	sz := image.Point{tsr.DimSize(1), tsr.DimSize(0)}
	esz := sz
	esz.X -= 2 * padWidth
	esz.Y -= 2 * padWidth
	var avg float32
	n := 0
	for y := 0; y < esz.Y; y++ {
		oy := y + padWidth
		avg += tsr.Value(oy, padWidth)
		avg += tsr.Value(oy, padWidth+esz.X-1)
	}
	n += 2 * esz.X
	for x := 0; x < esz.X; x++ {
		ox := x + padWidth
		avg += tsr.Value(padWidth, ox)
		avg += tsr.Value(padWidth+esz.Y-1, ox)
	}
	n += 2 * esz.X
	return avg / float32(n)
}

// FadePad fades given padding width of float32 image around sides
// gradually fading the edge value toward a mean edge value
func FadePad(tsr *tensor.Float32, padWidth int) {
	sz := image.Point{tsr.DimSize(1), tsr.DimSize(0)}
	usz := sz
	usz.Y -= padWidth
	usz.X -= padWidth
	avg := EdgeAvg(tsr, padWidth)
	for y := 0; y < sz.Y; y++ {
		var lv, rv float32
		switch {
		case y < padWidth:
			lv = tsr.Value(padWidth, padWidth)
			rv = tsr.Value(padWidth, usz.X-1)
		case y >= usz.Y:
			lv = tsr.Value(usz.Y-1, padWidth)
			rv = tsr.Value(usz.Y-1, usz.X-1)
		default:
			lv = tsr.Value(y, padWidth)
			rv = tsr.Value(y, usz.X-1)
		}
		for x := 0; x < padWidth; x++ {
			if y < x || y >= sz.Y-x {
				continue
			}
			p := float32(x) / float32(padWidth)
			pavg := (1 - p) * avg
			lpv := p*lv + pavg
			rpv := p*rv + pavg
			tsr.Set(lpv, y, x)
			tsr.Set(rpv, y, sz.X-1-x)
		}
	}
	for x := 0; x < sz.X; x++ {
		var tv, bv float32
		switch {
		case x < padWidth:
			tv = tsr.Value(padWidth, padWidth)
			bv = tsr.Value(usz.Y-1, padWidth)
		case x >= usz.X:
			tv = tsr.Value(padWidth, usz.X-1)
			bv = tsr.Value(usz.Y-1, usz.X-1)
		default:
			tv = tsr.Value(padWidth, x)
			bv = tsr.Value(usz.X-1, x)
		}
		for y := 0; y < padWidth; y++ {
			if x < y || x >= sz.X-y {
				continue
			}
			p := float32(y) / float32(padWidth)
			pavg := (1 - p) * avg
			tpv := p*tv + pavg
			bpv := p*bv + pavg
			tsr.Set(tpv, y, x)
			tsr.Set(bpv, sz.Y-1-y, x)
		}
	}
}

// FadePadRGB fades given padding width of float32 image around sides
// gradually fading the edge value toward a mean edge value.
// RGB version iterates over outer-most dimension of components.
func FadePadRGB(tsr *tensor.Float32, padWidth int) {
	nc := tsr.DimSize(0)
	for i := 0; i < nc; i++ {
		simg := tsr.SubSpace(i).(*tensor.Float32)
		FadePad(simg, padWidth)
	}
}
