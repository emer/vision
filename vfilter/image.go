// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"image/color"

	"goki.dev/colors"
	"goki.dev/etable/v2/etensor"
)

// RGBToTensor converts an RGB input image to an RGB etensor
// with outer dimension as RGB components.
// padWidth is the amount of padding to add on all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBToTensor(img image.Image, tsr *etensor.Float32, padWidth int, topZero bool) {
	bd := img.Bounds()
	sz := bd.Size()
	tsr.SetShape([]int{3, sz.Y + 2*padWidth, sz.X + 2*padWidth}, nil, []string{"RGB", "Y", "X"})
	for y := 0; y < sz.Y; y++ {
		for x := 0; x < sz.X; x++ {
			sy := y
			if !topZero {
				sy = (sz.Y - 1) - y
			}
			cv := img.At(bd.Min.X+x, bd.Min.Y+sy)
			r, g, b, _ := colors.ToFloat32(cv)
			tsr.Set([]int{0, y + padWidth, x + padWidth}, r)
			tsr.Set([]int{1, y + padWidth, x + padWidth}, g)
			tsr.Set([]int{2, y + padWidth, x + padWidth}, b)
		}
	}
}

// RGBTensorToImage converts an RGB etensor to image -- uses
// existing image if it is of correct size, otherwise makes a new one.
// etensor must have outer dimension as RGB components.
// padWidth is the amount of padding to subtract from all sides.
// topZero retains the Y=0 value at the top of the tensor --
// otherwise it is flipped with Y=0 at the bottom to be consistent
// with the emergent / OpenGL standard coordinate system
func RGBTensorToImage(img *image.RGBA, tsr *etensor.Float32, padWidth int, topZero bool) *image.RGBA {
	var sz image.Point
	sz.Y = tsr.Dim(1) - 2*padWidth
	sz.X = tsr.Dim(2) - 2*padWidth
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
			r := tsr.Value([]int{0, y + padWidth, x + padWidth})
			g := tsr.Value([]int{1, y + padWidth, x + padWidth})
			b := tsr.Value([]int{2, y + padWidth, x + padWidth})
			ri := uint8(r * 255)
			gi := uint8(g * 255)
			bi := uint8(b * 255)
			img.Set(x, sy, color.RGBA{ri, gi, bi, 255})
		}
	}
	return img
}

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
			r, g, b, _ := colors.ToFloat32(cv)
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

// WrapPad wraps given padding width of float32 image around sides
// i.e., padding for left side of image is the (mirrored) bits
// from the right side of image, etc.
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

// WrapPadRGB wraps given padding width of float32 image around sides
// i.e., padding for left side of image is the (mirrored) bits
// from the right side of image, etc.
// RGB version iterates over outer-most dimension of components.
func WrapPadRGB(tsr *etensor.Float32, padWidth int) {
	nc := tsr.Dim(0)
	for i := 0; i < nc; i++ {
		simg := tsr.SubSpace([]int{i}).(*etensor.Float32)
		WrapPad(simg, padWidth)
	}
}

// EdgeAvg returns the average value around the effective edge of image
// at padWidth in from each side
func EdgeAvg(tsr *etensor.Float32, padWidth int) float32 {
	sz := image.Point{tsr.Dim(1), tsr.Dim(0)}
	esz := sz
	esz.X -= 2 * padWidth
	esz.Y -= 2 * padWidth
	var avg float32
	n := 0
	for y := 0; y < esz.Y; y++ {
		oy := y + padWidth
		avg += tsr.Value([]int{oy, padWidth})
		avg += tsr.Value([]int{oy, padWidth + esz.X - 1})
	}
	n += 2 * esz.X
	for x := 0; x < esz.X; x++ {
		ox := x + padWidth
		avg += tsr.Value([]int{padWidth, ox})
		avg += tsr.Value([]int{padWidth + esz.Y - 1, ox})
	}
	n += 2 * esz.X
	return avg / float32(n)
}

// FadePad fades given padding width of float32 image around sides
// gradually fading the edge value toward a mean edge value
func FadePad(tsr *etensor.Float32, padWidth int) {
	sz := image.Point{tsr.Dim(1), tsr.Dim(0)}
	usz := sz
	usz.Y -= padWidth
	usz.X -= padWidth
	avg := EdgeAvg(tsr, padWidth)
	for y := 0; y < sz.Y; y++ {
		var lv, rv float32
		switch {
		case y < padWidth:
			lv = tsr.Value([]int{padWidth, padWidth})
			rv = tsr.Value([]int{padWidth, usz.X - 1})
		case y >= usz.Y:
			lv = tsr.Value([]int{usz.Y - 1, padWidth})
			rv = tsr.Value([]int{usz.Y - 1, usz.X - 1})
		default:
			lv = tsr.Value([]int{y, padWidth})
			rv = tsr.Value([]int{y, usz.X - 1})
		}
		for x := 0; x < padWidth; x++ {
			if y < x || y >= sz.Y-x {
				continue
			}
			p := float32(x) / float32(padWidth)
			pavg := (1 - p) * avg
			lpv := p*lv + pavg
			rpv := p*rv + pavg
			tsr.Set([]int{y, x}, lpv)
			tsr.Set([]int{y, sz.X - 1 - x}, rpv)
		}
	}
	for x := 0; x < sz.X; x++ {
		var tv, bv float32
		switch {
		case x < padWidth:
			tv = tsr.Value([]int{padWidth, padWidth})
			bv = tsr.Value([]int{usz.Y - 1, padWidth})
		case x >= usz.X:
			tv = tsr.Value([]int{padWidth, usz.X - 1})
			bv = tsr.Value([]int{usz.Y - 1, usz.X - 1})
		default:
			tv = tsr.Value([]int{padWidth, x})
			bv = tsr.Value([]int{usz.X - 1, x})
		}
		for y := 0; y < padWidth; y++ {
			if x < y || x >= sz.X-y {
				continue
			}
			p := float32(y) / float32(padWidth)
			pavg := (1 - p) * avg
			tpv := p*tv + pavg
			bpv := p*bv + pavg
			tsr.Set([]int{y, x}, tpv)
			tsr.Set([]int{sz.Y - 1 - y, x}, bpv)
		}
	}
}

// FadePadRGB fades given padding width of float32 image around sides
// gradually fading the edge value toward a mean edge value.
// RGB version iterates over outer-most dimension of components.
func FadePadRGB(tsr *etensor.Float32, padWidth int) {
	nc := tsr.Dim(0)
	for i := 0; i < nc; i++ {
		simg := tsr.SubSpace([]int{i}).(*etensor.Float32)
		FadePad(simg, padWidth)
	}
}
