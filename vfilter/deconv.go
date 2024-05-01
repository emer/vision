// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"log"

	"cogentcore.org/core/tensor"
)

// Deconv performs reverse convolution of filter -- given output of filter,
// accumulates an input image as sum of filter * output activation.
// img *must* have border (padding) so that filters are
// applied without any bounds checking -- wrapping etc is all
// done in the padding process, which is much more efficient.
// img must be a 2D tensor of image values (convert RGB to grey first).
// Everything must be organized row major as tensor default.
// Out shape dims are: Y, X, Polarity (2), Angle
// where the 2 polarities (on, off) are for positive and and
// negative filter values, respectively.
func Deconv(geom *Geom, flt *tensor.Float32, img, out *tensor.Float32, gain float32) {
	nf := flt.DimSize(0)
	fy := flt.DimSize(1)
	fx := flt.DimSize(2)

	geom.FiltSz = image.Point{fx, fy}
	geom.UpdtFilt()

	imgSz := image.Point{img.DimSize(1), img.DimSize(0)}
	geom.SetSize(imgSz)
	oshp := []int{int(geom.Out.Y), int(geom.Out.X), 2, nf}
	if !tensor.EqualInts(oshp, out.Shp) {
		log.Printf("Deconv output shape not correct for input\n")
		return
	}
	ist := geom.Border.Sub(geom.FiltLt)
	fsz := fx * fy
	for f := 0; f < nf; f++ {
		fst := f * fsz
		for y := 0; y < geom.Out.Y; y++ {
			iy := int(ist.Y + y*geom.Spacing.Y)
			for x := 0; x < geom.Out.X; x++ {
				ix := ist.X + x*geom.Spacing.X
				act := float32(0)
				if av := out.Value([]int{y, x, 1, f}); av > 0 {
					act = av
				} else {
					act = -out.Value([]int{y, x, 0, f})
				}
				fi := 0
				for fy := 0; fy < geom.FiltSz.Y; fy++ {
					for fx := 0; fx < geom.FiltSz.X; fx++ {
						fv := flt.Values[fst+fi]
						iv := act * fv
						iv += img.Value([]int{iy + fy, ix + fx})
						img.Set([]int{iy + fy, ix + fx}, iv)
						fi++
					}
				}
			}
		}
	}
}
