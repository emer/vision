// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package dog provides the Difference-of-Gaussians (DoG) filter for visual and other
forms of signal processing
*/
package dog

import (
	"goki.dev/etable/v2/etable"
	"goki.dev/etable/v2/etensor"
	"goki.dev/mat32/v2"
)

// dog.Filter specifies a DoG Difference of Gaussians filter function.
type Filter struct {

	// is this filter active?
	On bool

	// how much relative weight does this filter have when combined with other filters
	Wt float32 `viewif:"On"`

	// overall gain multiplier applied after dog filtering -- only relevant if not using renormalization (otherwize it just gets renormed away)
	Gain float32 `viewif:"On" def:"8"`

	// gain for the on component of filter, only relevant for color-opponent DoG's
	OnGain float32 `viewif:"On" def:"1"`

	// size of the overall filter -- number of pixels wide and tall for a square matrix used to encode the filter -- filter is centered within this square -- typically an even number, min effective size ~6
	Size int `viewif:"On"`

	// how far apart to space the centers of the dog filters -- 1 = every pixel, 2 = every other pixel, etc -- high-res should be 1 or 2, lower res can be increments therefrom
	Spacing int `viewif:"On"`

	// gaussian sigma for the narrower On gaussian, in normalized units relative to Size
	OnSig float32 `viewif:"On" def:"0.125"`

	// gaussian sigma for the wider Off gaussian, in normalized units relative to Size
	OffSig float32 `viewif:"On" def:"0.25"`

	// cut off the filter (to zero) outside a circle of diameter = Size -- makes the filter more radially symmetric
	CircleEdge bool `viewif:"On" def:"true"`
}

func (gf *Filter) Defaults() {
	gf.On = true
	gf.Wt = 1
	gf.Gain = 8
	gf.OnGain = 1
	gf.Size = 12
	gf.Spacing = 2
	gf.OnSig = 0.125
	gf.OffSig = 0.25
	gf.CircleEdge = true
}

func (gf *Filter) Update() {
}

// SetSize sets the size and spacing -- these are the main params
// that need to be varied for standard V1 dogs.
func (gf *Filter) SetSize(sz, spc int) {
	gf.Size = sz
	gf.Spacing = spc
}

// GaussDenSig returns gaussian density for given value and sigma
func GaussDenSig(x, sig float32) float32 {
	x /= sig
	return 0.398942280 * mat32.Exp(-0.5*x*x) / sig
}

// ToTensor renders dog filters into the given etable etensor.Tensor,
// setting dimensions to [3][Y][X] where Y = X = Size, and
// first one is On-filter, second is Off-filter, and third is Net On - Off
func (gf *Filter) ToTensor(tsr *etensor.Float32) {
	tsr.SetShape([]int{int(FiltersN), gf.Size, gf.Size}, nil, []string{"3", "Y", "X"})

	ctr := 0.5 * float32(gf.Size-1)
	radius := float32(gf.Size) * 0.5

	gsOn := gf.OnSig * float32(gf.Size)
	gsOff := gf.OffSig * float32(gf.Size)

	var posSum, negSum, onSum, offSum float32
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			xf := float32(x) - ctr
			yf := float32(y) - ctr

			dist := mat32.Hypot(xf, yf)
			var ong, offg float32
			if !(gf.CircleEdge && (dist > radius)) {
				ong = GaussDenSig(dist, gsOn)
				offg = GaussDenSig(dist, gsOff)
			}
			tsr.Set([]int{int(On), y, x}, ong)
			tsr.Set([]int{int(Off), y, x}, offg)
			onSum += ong
			offSum += offg
			net := ong - offg
			tsr.Set([]int{int(Net), y, x}, net)
			if net > 0 {
				posSum += net
			} else if net < 0 {
				negSum += -net
			}
		}
	}
	// renorm each half, separate components
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			val := tsr.Value([]int{int(Net), y, x})
			if val > 0 {
				val /= posSum
			} else if val < 0 {
				val /= negSum
			}
			tsr.Set([]int{int(Net), y, x}, val)
			on := tsr.Value([]int{int(On), y, x})
			tsr.Set([]int{int(On), y, x}, on/onSum)
			off := tsr.Value([]int{int(Off), y, x})
			tsr.Set([]int{int(Off), y, x}, off/offSum)
		}
	}
}

// ToTable renders filters into the given etable.Table
// setting a column named Version and  a column named Filter
// to the filter for that version (on, off, net)
// This is useful for display and validation purposes.
func (gf *Filter) ToTable(tab *etable.Table) {
	tab.SetFromSchema(etable.Schema{
		{"Version", etensor.STRING, nil, nil},
		{"Filter", etensor.FLOAT32, []int{int(FiltersN), gf.Size, gf.Size}, []string{"Version", "Y", "X"}},
	}, 3)
	gf.ToTensor(tab.Cols[1].(*etensor.Float32))
	tab.SetCellStringIdx(0, int(On), "On")
	tab.SetCellStringIdx(0, int(Off), "Off")
	tab.SetCellStringIdx(0, int(Net), "Net")
}

// FilterTensor extracts the given filter subspace from set of 3 filters in input tensor
// 0 = On, 1 = Off, 2 = Net
func (gf *Filter) FilterTensor(tsr *etensor.Float32, filt Filters) *etensor.Float32 {
	return tsr.SubSpace([]int{int(filt)}).(*etensor.Float32)
}

// Filters is the type of filter
type Filters int

const (
	On Filters = iota
	Off
	Net
	FiltersN
)
