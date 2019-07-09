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
func Conv(geom *Geom, flt *etensor.Float32, img, out *etensor.Float32) {
	nf := flt.Dim(0)
	fy := int32(flt.Dim(1))
	fx := int32(flt.Dim(2))

	geom.FiltSz = mat32.Vec2i{fx, fy}
	geom.UpdtFilt()

	imgSz := mat32.Vec2i{int32(img.Dim(1)), int32(img.Dim(0))}
	geom.SetSize(imgSz)
	oshp := []int{nf, 2, int(geom.Out.Y), int(geom.Out.X)}
	if !etensor.EqualInts(oshp, out.Shp) {
		out.SetShape(oshp, nil, []string{"Filter", "Pol", "Y", "X"})
	}
	var wg sync.WaitGroup
	for f := 0; f < nf; f++ {
		wg.Add(1)
		go ConvFlt(&wg, geom, f, flt, img, out)
	}
	wg.Wait()
}

// ConvFlt performs convolution using given filter over entire image
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
				out.Set([]int{fno, 0, y, x}, sum)
				out.Set([]int{fno, 1, y, x}, float32(0))
			} else {
				out.Set([]int{fno, 0, y, x}, float32(0))
				out.Set([]int{fno, 1, y, x}, sum)
			}
		}
	}

	wg.Done()
}
