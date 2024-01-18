// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

//go:generate core generate -add-types

import (
	"log"

	"cogentcore.org/core/mat32"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/vision/v2/fffb"
	"github.com/emer/vision/v2/nxx1"
)

// KWTA contains all the parameters needed for computing FFFB
// (feedforward & feedback) inhibition that results in roughly
// k-Winner-Take-All behavior.
type KWTA struct {

	// whether to run kWTA or not
	On bool

	// maximum number of iterations to perform
	Iters int

	// threshold on delta-activation (change in activation) for stopping updating of activations
	DelActThr float32 `def:"0.005"`

	// layer-level feedforward & feedback inhibition -- applied over entire set of values
	LayFFFB fffb.Params `view:"inline"`

	// pool-level (feature groups) feedforward and feedback inhibition -- applied within inner-most dimensions inside outer 2 dimensions (if Pool method is called)
	PoolFFFB fffb.Params `view:"inline"`

	// Noisy X/X+1 rate code activation function parameters
	XX1 nxx1.Params `view:"inline"`

	// time constant for integrating activation
	ActTau float32 `def:"3"`

	// maximal conductances levels for channels
	Gbar Chans `view:"inline"`

	// reversal potentials for each channel
	Erev Chans `view:"inline"`

	// Erev - Act.Thr for each channel -- used in computing GeThrFmG among others
	ErevSubThr Chans `inactive:"+" view:"-"`

	// Act.Thr - Erev for each channel -- used in computing GeThrFmG among others
	ThrSubErev Chans `inactive:"+" view:"-" json:"-" xml:"-"`

	//
	ActDt float32 `view:"-"; json"-" xml"-" desc:"integration rate = 1/ tau"`
}

func (kwta *KWTA) Defaults() {
	kwta.On = true
	kwta.Iters = 20
	kwta.DelActThr = 0.005
	kwta.LayFFFB.Defaults()
	kwta.PoolFFFB.Defaults()
	kwta.LayFFFB.On = true
	kwta.PoolFFFB.On = true
	kwta.PoolFFFB.Gi = 2.0
	kwta.XX1.Defaults()
	kwta.ActTau = 3
	kwta.Gbar.SetAll(0.5, 0.1, 1.0, 1.0) // 0.5 is key for 1.0 inputs
	kwta.Erev.SetAll(1.0, 0.3, 0.3, 0.1)
	kwta.Update()
}

// Update must be called after any changes to parameters
func (kwta *KWTA) Update() {
	kwta.LayFFFB.Update()
	kwta.PoolFFFB.Update()
	kwta.XX1.Update()
	kwta.ErevSubThr.SetFmOtherMinus(kwta.Erev, kwta.XX1.Thr)
	kwta.ThrSubErev.SetFmMinusOther(kwta.XX1.Thr, kwta.Erev)
	kwta.ActDt = 1 / kwta.ActTau
}

// GeThrFmG computes the threshold for Ge based on other conductances
func (kwta *KWTA) GeThrFmG(gi float32) float32 {
	ge := ((kwta.Gbar.I*gi*kwta.ErevSubThr.I + kwta.Gbar.L*kwta.ErevSubThr.L) / kwta.ThrSubErev.E)
	return ge
}

// ActFmG computes rate-coded activation Act from conductances Ge and Gi
func (kwta *KWTA) ActFmG(geThr, ge, act float32) (nwAct, delAct float32) {
	nwAct = kwta.XX1.NoisyXX1(ge*kwta.Gbar.E - geThr)
	delAct = kwta.ActDt * (nwAct - act)
	nwAct = act + delAct
	return nwAct, delAct
}

// KWTALayer computes k-Winner-Take-All activation values from raw inputs.
// act output tensor is set to same shape as raw inputs if not already.
// This version just computes a "layer" level of inhibition across the
// entire set of tensor values.
// extGi is extra / external Gi inhibition per unit
// -- e.g. from neighbor inhib -- must be size of raw, act.
func (kwta *KWTA) KWTALayer(raw, act, extGi *etensor.Float32) {
	inhib := fffb.Inhib{}
	raws := raw.Values // these are ge

	if !act.Shape.IsEqual(&raw.Shape) {
		act.SetShape(raw.Shape.Shp, raw.Shape.Strd, raw.Shape.Nms)
	}
	if extGi != nil {
		if !extGi.Shape.IsEqual(&raw.Shape) {
			log.Println("KWTALayer: extGi is not correct shape, will not be used!")
			extGi = nil
		}
	}

	acts := act.Values

	inhib.Ge.Init()
	for i, ge := range raws {
		inhib.Ge.UpdateVal(ge, int32(i))
	}
	inhib.Ge.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(&inhib)
		inhib.Act.Init()
		maxDelAct := float32(0)
		for i := range acts {
			gi := inhib.Gi
			if extGi != nil {
				gi += extGi.Values[i]
			}
			geThr := kwta.GeThrFmG(gi)
			ge := raws[i]
			nwAct, delAct := kwta.ActFmG(geThr, ge, acts[i])
			maxDelAct = mat32.Max(maxDelAct, mat32.Abs(delAct))
			inhib.Act.UpdateVal(nwAct, int32(i))
			acts[i] = nwAct
		}
		inhib.Act.CalcAvg()
		if cy > 2 && maxDelAct < kwta.DelActThr {
			break
		}
	}
}

// KWTAPool computes k-Winner-Take-All activation values from raw inputs
// act output tensor is set to same shape as raw inputs if not already.
// This version computes both Layer and Pool (feature-group) level
// inibition -- tensors must be 4 dimensional -- outer 2D is Y, X Layer
// and inner 2D are features (pools) per location.
// The inhib slice is required for pool-level inhibition and will
// be automatically sized to outer X,Y dims if not big enough.
// For best performance store this and reuse to avoid memory allocations.
// extGi is extra / external Gi inhibition per unit
// -- e.g. from neighbor inhib -- must be size of raw, act.
func (kwta *KWTA) KWTAPool(raw, act *etensor.Float32, inhib *fffb.Inhibs, extGi *etensor.Float32) {
	layInhib := fffb.Inhib{}

	raws := raw.Values // these are ge

	if !act.Shape.IsEqual(&raw.Shape) {
		act.SetShape(raw.Shape.Shp, raw.Shape.Strd, raw.Shape.Nms)
	}
	if extGi != nil {
		if !extGi.Shape.IsEqual(&raw.Shape) {
			log.Println("KWTAPool: extGi is not correct shape, will not be used!")
			extGi = nil
		}
	}

	acts := act.Values

	layY := raw.Dim(0)
	layX := raw.Dim(1)
	layN := layY * layX

	plY := raw.Dim(2)
	plX := raw.Dim(3)
	plN := plY * plX

	if len(*inhib) < layN {
		if cap(*inhib) < layN {
			*inhib = make([]fffb.Inhib, layN)
		} else {
			*inhib = (*inhib)[0:layN]
		}
	}

	layInhib.Ge.Init()
	pi := 0
	for ly := 0; ly < layY; ly++ {
		for lx := 0; lx < layX; lx++ {
			plInhib := &((*inhib)[pi])
			plInhib.Ge.Init()
			pui := pi * plN
			ui := 0
			for py := 0; py < plY; py++ {
				for px := 0; px < plX; px++ {
					idx := pui + ui
					ge := raws[idx]
					layInhib.Ge.UpdateVal(ge, int32(idx))
					plInhib.Ge.UpdateVal(ge, int32(ui))
					ui++
				}
			}
			plInhib.Ge.CalcAvg()
			pi++
		}
	}
	layInhib.Ge.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(&layInhib)

		layInhib.Act.Init()
		maxDelAct := float32(0)
		pi := 0
		for ly := 0; ly < layY; ly++ {
			for lx := 0; lx < layX; lx++ {
				plInhib := &((*inhib)[pi])

				kwta.PoolFFFB.Inhib(plInhib)

				giPool := mat32.Max(layInhib.Gi, plInhib.Gi)

				plInhib.Act.Init()
				pui := pi * plN
				ui := 0
				for py := 0; py < plY; py++ {
					for px := 0; px < plX; px++ {
						idx := pui + ui
						gi := giPool
						if extGi != nil {
							eIn := extGi.Values[idx]
							eGi := kwta.PoolFFFB.Gi * kwta.PoolFFFB.FFInhib(eIn, eIn)
							gi = mat32.Max(gi, eGi)
						}
						geThr := kwta.GeThrFmG(gi)
						ge := raws[idx]
						act := acts[idx]
						nwAct, delAct := kwta.ActFmG(geThr, ge, act)
						maxDelAct = mat32.Max(maxDelAct, mat32.Abs(delAct))
						layInhib.Act.UpdateVal(nwAct, int32(idx))
						plInhib.Act.UpdateVal(nwAct, int32(ui))
						acts[idx] = nwAct

						ui++
					}
				}
				plInhib.Act.CalcAvg()
				pi++
			}
		}
		layInhib.Act.CalcAvg()
		if cy > 2 && maxDelAct < kwta.DelActThr {
			// fmt.Printf("under thr at cycle: %v\n", cy)
			break
		}
	}
}
