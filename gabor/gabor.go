// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package gabor provides a gabor filter for visual and other
forms of signal processing
*/
package gabor

//go:generate core generate -add-types

import (
	"math"

	"cogentcore.org/core/mat32"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
)

// gabor.Filter specifies a gabor filter function,
// i.e., a 2d Gaussian envelope times a sinusoidal plane wave.
// By default it produces 2 phase asymmetric edge detector filters.
type Filter struct {

	// is this filter active?
	On bool

	// how much relative weight does this filter have when combined with other filters
	Wt float32 `viewif:"On"`

	// overall gain multiplier applied after filtering -- only relevant if not using renormalization (otherwize it just gets renormed away)
	Gain float32 `viewif:"On" def:"2"`

	// size of the overall filter -- number of pixels wide and tall for a square matrix used to encode the filter -- filter is centered within this square -- typically an even number, min effective size ~6
	Size int `viewif:"On"`

	// wavelength of the sine waves -- number of pixels over which a full period of the wave takes place -- typically same as Size (computation adds a 2 PI factor to translate into pixels instead of radians)
	WvLen float32 `viewif:"On"`

	// how far apart to space the centers of the gabor filters -- 1 = every pixel, 2 = every other pixel, etc -- high-res should be 1 or 2, lower res can be increments therefrom
	Spacing int `viewif:"On"`

	// gaussian sigma for the length dimension (elongated axis perpendicular to the sine waves) -- as a normalized proportion of filter Size
	SigLen float32 `viewif:"On" def:"0.3"`

	// gaussian sigma for the width dimension (in the direction of the sine waves) -- as a normalized proportion of filter size
	SigWd float32 `viewif:"On" def:"0.15,0.2"`

	// phase offset for the sine wave, in degrees -- 0 = asymmetric sine wave, 90 = symmetric cosine wave
	Phase float32 `viewif:"On" def:"0,90"`

	// cut off the filter (to zero) outside a circle of diameter = Size -- makes the filter more radially symmetric
	CircleEdge bool `viewif:"On" def:"true"`

	// number of different angles of overall gabor filter orientation to use -- first angle is always horizontal
	NAngles int `viewif:"On" def:"4"`
}

func (gf *Filter) Defaults() {
	gf.On = true
	gf.Wt = 1
	gf.Gain = 2
	gf.Size = 6
	gf.Spacing = 2
	gf.WvLen = 6
	gf.SigLen = 0.3
	gf.SigWd = 0.2
	gf.Phase = 0
	gf.CircleEdge = true
	gf.NAngles = 4
}

func (gf *Filter) Update() {
}

// SetSize sets the size and WvLen to same value, and also sets spacing
// these are the main params that need to be varied for standard V1 gabors
func (gf *Filter) SetSize(sz, spc int) {
	gf.Size = sz
	gf.WvLen = float32(sz)
	gf.Spacing = spc
}

// ToTensor renders filters into the given etable etensor.Tensor,
// setting dimensions to [angle][Y][X] where Y = X = Size
func (gf *Filter) ToTensor(tsr *etensor.Float32) {
	tsr.SetShape([]int{gf.NAngles, gf.Size, gf.Size}, nil, []string{"Angles", "Y", "X"})

	ctr := 0.5 * float32(gf.Size-1)
	angInc := math.Pi / float32(gf.NAngles)

	radius := float32(gf.Size) * 0.5

	gsLen := gf.SigLen * float32(gf.Size)
	gsWd := gf.SigWd * float32(gf.Size)

	lenNorm := 1.0 / (2.0 * gsLen * gsLen)
	wdNorm := 1.0 / (2.0 * gsWd * gsWd)

	twoPiNorm := (2.0 * math.Pi) / gf.WvLen
	phsRad := mat32.DegToRad(gf.Phase)

	for ang := 0; ang < gf.NAngles; ang++ {
		angf := -float32(ang) * angInc

		posSum := float32(0)
		negSum := float32(0)
		for x := 0; x < gf.Size; x++ {
			for y := 0; y < gf.Size; y++ {
				xf := float32(x) - ctr
				yf := float32(y) - ctr

				dist := mat32.Hypot(xf, yf)
				val := float32(0)
				if !(gf.CircleEdge && (dist > radius)) {
					nx := xf*mat32.Cos(angf) - yf*mat32.Sin(angf)
					ny := yf*mat32.Cos(angf) + xf*mat32.Sin(angf)
					gauss := mat32.Exp(-(lenNorm*(nx*nx) + wdNorm*(ny*ny)))
					sin := mat32.Sin(twoPiNorm*ny + phsRad)
					val = gauss * sin
					if val > 0 {
						posSum += val
					} else if val < 0 {
						negSum += -val
					}
				}
				tsr.Set([]int{ang, y, x}, val)
			}
		}
		// renorm each half
		posNorm := float32(1) / posSum
		negNorm := float32(1) / negSum
		for x := 0; x < gf.Size; x++ {
			for y := 0; y < gf.Size; y++ {
				val := tsr.Value([]int{ang, y, x})
				if val > 0 {
					val *= posNorm
				} else if val < 0 {
					val *= negNorm
				}
				tsr.Set([]int{ang, y, x}, val)
			}
		}
	}
}

// ToTable renders filters into the given etable.Table
// setting a column named Angle to the angle and
// a column named Gabor to the filter for that angle.
// This is useful for display and validation purposes.
func (gf *Filter) ToTable(tab *etable.Table) {
	tab.SetFromSchema(etable.Schema{
		{"Angle", etensor.FLOAT32, nil, nil},
		{"Filter", etensor.FLOAT32, []int{gf.NAngles, gf.Size, gf.Size}, []string{"Angle", "Y", "X"}},
	}, gf.NAngles)
	gf.ToTensor(tab.Cols[1].(*etensor.Float32))
	angInc := math.Pi / float32(gf.NAngles)
	for ang := 0; ang < gf.NAngles; ang++ {
		angf := mat32.RadToDeg(-float32(ang) * angInc)
		tab.SetCellFloatIdx(0, ang, float64(-angf))
	}
}
