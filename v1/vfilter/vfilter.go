// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package vfilter provides filtering methods for the vision package.
These apply a given etensor.Tensor filter set to a 2D visual input.
*/
package vfilter

import (
	"sync"

	"github.com/emer/etable/etensor"
	"github.com/goki/gi/mat32"
)

// Conv performs convolution of filter over img into out
// Computation is parallel in number of different filter types
// (outer dim of flt) as that will be most memory efficient.
// img must be a 2D tensor of image values (convert color to grey first).
// everything must be organized row major as etensor default.
// Out shape dims are: Y, X, Polarity (2), Angle
// where the 2 polarities (on, off) are for positive and and
// negative filter values, respectively.
func Conv(geom *Geom, flt *etensor.Float32, img, out *etensor.Float32) {
	nf := flt.Dim(0)
	fy := int32(flt.Dim(1))
	fx := int32(flt.Dim(2))

	geom.FiltSz = mat32.Vec2i{fx, fy}
	geom.UpdtFilt()

	imgSz := mat32.Vec2i{int32(img.Dim(1)), int32(img.Dim(0))}
	geom.SetSize(imgSz)
	oshp := []int{int(geom.Out.Y), int(geom.Out.X), 2, nf}
	if !etensor.EqualInts(oshp, out.Shp) {
		out.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	var wg sync.WaitGroup
	for f := 0; f < nf; f++ {
		wg.Add(1)
		go ConvFlt(&wg, geom, f, flt, img, out)
	}
	wg.Wait()
}

// ConvFlt performs convolution using given filter over entire image
// This is called by Conv using different parallel goroutines
func ConvFlt(wg *sync.WaitGroup, geom *Geom, fno int, flt *etensor.Float32, img, out *etensor.Float32) {
	fst := fno * int(geom.FiltSz.Y) * int(geom.FiltSz.X)
	ist := geom.Border.Sub(geom.FiltLt)
	for y := 0; y < int(geom.Out.Y); y++ {
		iy := int(ist.Y + int32(y)*geom.Spacing.Y)
		for x := 0; x < int(geom.Out.X); x++ {
			ix := int(ist.X + int32(x)*geom.Spacing.X)
			sum := float32(0)
			fi := 0
			for fy := 0; fy < int(geom.FiltSz.Y); fy++ {
				for fx := 0; fx < int(geom.FiltSz.X); fx++ {
					iv := img.Value([]int{iy + fy, ix + fx})
					fv := flt.Values[fst+fi]
					sum += iv * fv
					fi++
				}
			}
			if sum > 0 {
				out.Set([]int{y, x, 0, fno}, sum)
				out.Set([]int{y, x, 1, fno}, float32(0))
			} else {
				out.Set([]int{y, x, 0, fno}, float32(0))
				out.Set([]int{y, x, 1, fno}, -sum)
			}
		}
	}
	wg.Done()
}

// MaxPool performs max-pooling over given pool size,
// sensitive to the feature structure of the input, which
// must have shape: Y, X, Polarities, Angles
func MaxPool(psize mat32.Vec2i, in, out *etensor.Float32) {
	ny := in.Dim(0)
	nx := in.Dim(1)
	pol := in.Dim(2)
	nang := in.Dim(3)
	oy := ny / int(psize.Y)
	ox := nx / int(psize.X)

	oshp := []int{oy, ox, pol, nang}
	if !etensor.EqualInts(oshp, out.Shp) {
		out.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	nin := pol * nang
	var wg sync.WaitGroup
	for f := 0; f < nin; f++ {
		wg.Add(1)
		go MaxPoolOne(&wg, f, psize, in, out)
	}
	wg.Wait()
}

// MaxPoolOne performs max pooling for one specific polarity, angle
// called as a goroutine from MaxPool
func MaxPoolOne(wg *sync.WaitGroup, fno int, psize mat32.Vec2i, in, out *etensor.Float32) {
	ny := out.Dim(0)
	nx := out.Dim(1)
	nang := out.Dim(3)
	pol := fno / nang
	ang := fno % nang
	for y := 0; y < ny; y++ {
		iy := y * int(psize.Y)
		for x := 0; x < nx; x++ {
			ix := x * int(psize.X)
			max := float32(0)
			for py := 0; py < int(psize.Y); py++ {
				for px := 0; px < int(psize.X); px++ {
					iv := in.Value([]int{iy + py, ix + px, pol, ang})
					if iv > max {
						max = iv
					}
				}
			}
			out.Set([]int{y, x, pol, ang}, max)
		}
	}
	wg.Done()
}
