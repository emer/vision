// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfilter

import (
	"image"
	"sync"

	"cogentcore.org/core/tensor"
	"github.com/emer/vision/v2/nproc"
)

// Conv performs convolution of filter over img into out.
// img *must* have border (padding) so that filters are
// applied without any bounds checking -- wrapping etc is all
// done in the padding process, which is much more efficient.
// Computation is parallel in number of different filter types
// (outer dim of flt) as that will be most memory efficient.
// img must be a 2D tensor of image values (convert RGB to grey first).
// Everything must be organized row major as tensor default.
// Out shape dims are: Y, X, Polarity (2), Angle
// where the 2 polarities (on, off) are for positive and and
// negative filter values, respectively.
func Conv(geom *Geom, flt *tensor.Float32, img, out *tensor.Float32, gain float32) {
	nf := flt.DimSize(0)
	fy := flt.DimSize(1)
	fx := flt.DimSize(2)

	geom.FiltSz = image.Point{fx, fy}
	geom.UpdtFilt()

	imgSz := image.Point{img.DimSize(1), img.DimSize(0)}
	geom.SetSize(imgSz)
	out.SetShapeSizes(int(geom.Out.Y), int(geom.Out.X), 2, nf)
	ncpu := nproc.NumCPU()
	nthrs, nper, rmdr := nproc.ThreadNs(ncpu, nf)
	var wg sync.WaitGroup
	for th := 0; th < nthrs; th++ {
		wg.Add(1)
		f := th * nper
		go convThr(&wg, geom, f, nper, flt, img, out, gain)
	}
	if rmdr > 0 {
		wg.Add(1)
		f := nthrs * nper
		go convThr(&wg, geom, f, rmdr, flt, img, out, gain)
	}
	wg.Wait()
}

// convThr is per-thread implementation
func convThr(wg *sync.WaitGroup, geom *Geom, fno, nf int, flt *tensor.Float32, img, out *tensor.Float32, gain float32) {
	ist := geom.Border.Sub(geom.FiltLt)
	fsz := int(geom.FiltSz.Y) * int(geom.FiltSz.X)
	for fi := 0; fi < nf; fi++ {
		f := fno + fi
		fst := f * fsz
		for y := 0; y < geom.Out.Y; y++ {
			iy := int(ist.Y + y*geom.Spacing.Y)
			for x := 0; x < geom.Out.X; x++ {
				ix := ist.X + x*geom.Spacing.X
				sum := float32(0)
				fi := 0
				for fy := 0; fy < geom.FiltSz.Y; fy++ {
					for fx := 0; fx < geom.FiltSz.X; fx++ {
						iv := img.Value(iy+fy, ix+fx)
						fv := flt.Values[fst+fi]
						sum += iv * fv
						fi++
					}
				}
				sum *= gain
				if sum > 0 {
					out.Set(sum, y, x, 0, f)
					out.Set(float32(0), y, x, 1, f)
				} else {
					out.Set(float32(0), y, x, 0, f)
					out.Set(-sum, y, x, 1, f)
				}
			}
		}
	}
	wg.Done()
}
