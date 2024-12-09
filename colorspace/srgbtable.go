// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colorspace

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
)

// SRGBToOp implements a lookup-table for the conversion of
// SRGB components to LMS color opponent values.
// After all this, it looks like the direct computation is faster
// than the lookup table!  In any case, it is all here and reasonably
// accurate (mostly under 1.0e-4 according to testing)
type SRGBToOp struct {

	// number of levels in the lookup table -- linear interpolation used
	Levels int

	// lookup table
	Table tensor.Float32
}

// TheSRGBToOp is the instance of SRGBToOp to use
var TheSRGBToOp SRGBToOp

// Init does initialization if not yet initialized
func (so *SRGBToOp) Init() {
	if so.Levels != 0 {
		return
	}

	so.Levels = 64
	ll := so.Levels
	llf := float32(ll)
	so.Table.SetShapeSizes(int(LMSComponentsN), so.Levels, so.Levels, so.Levels)
	// fmt.Printf("table size: %d\n", so.Table.Len())
	for bi := 0; bi < ll; bi++ {
		bf := float32(bi) / llf
		for gi := 0; gi < ll; gi++ {
			gf := float32(gi) / llf
			for ri := 0; ri < ll; ri++ {
				rf := float32(ri) / llf
				lc, mc, sc, lmc, lvm, svlm, grey := SRGBToLMSComps(rf, gf, bf)
				so.Table.Set(lc, int(LC), ri, gi, bi)
				so.Table.Set(mc, int(MC), ri, gi, bi)
				so.Table.Set(sc, int(SC), ri, gi, bi)
				so.Table.Set(lmc, int(LMC), ri, gi, bi)
				so.Table.Set(lvm, int(LvMC), ri, gi, bi)
				so.Table.Set(svlm, int(SvLMC), ri, gi, bi)
				so.Table.Set(grey, int(GREY), ri, gi, bi)
			}
		}
	}
}

func (so *SRGBToOp) InterpIdx(val float32) (loi, hii int, pctlo, pcthi float32) {
	fi := val * float32(so.Levels)
	loi = int(math32.Floor(fi))
	hii = int(math32.Ceil(fi))
	switch {
	case loi == so.Levels-1:
		hii = loi
		loi = loi - 1
		pcthi = 1
		pctlo = 0
	case hii == loi:
		hii = loi + 1
		pcthi = 0.0
		pctlo = 1.0
	default:
		pcthi = fi - float32(loi)
		pctlo = 1 - pcthi

	}
	return
}

func (so *SRGBToOp) Lookup(r, g, b float32) (lc, mc, sc, lmc, lvm, svlm, grey float32) {
	so.Init()
	var tmp [LMSComponentsN]float32

	r0, r1, r0p, r1p := so.InterpIdx(r)
	g0, g1, g0p, g1p := so.InterpIdx(g)
	b0, b1, b0p, b1p := so.InterpIdx(b)

	for i := 0; i < int(LMSComponentsN); i++ {
		c00 := so.Table.Value(i, r0, g0, b0)*r0p +
			so.Table.Value(i, r1, g0, b0)*r1p
		c01 := so.Table.Value(i, r0, g0, b1)*r0p +
			so.Table.Value(i, r1, g0, b1)*r1p
		c10 := so.Table.Value(i, r0, g1, b0)*r0p +
			so.Table.Value(i, r1, g1, b0)*r1p
		c11 := so.Table.Value(i, r0, g1, b1)*r0p +
			so.Table.Value(i, r1, g1, b1)*r1p
		c0 := c00*g0p + c10*g1p
		c1 := c01*g0p + c11*g1p
		c := c0*b0p + c1*b1p
		tmp[i] = c
	}
	lc = tmp[LC]
	mc = tmp[MC]
	sc = tmp[SC]
	lmc = tmp[LMC]
	lvm = tmp[LvMC]
	svlm = tmp[SvLMC]
	grey = tmp[GREY]
	return
}
