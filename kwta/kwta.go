// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

import (
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/mat32"
)

//////////////////////////////////////////////////////////////////////////////////////
//  From emer.leabra.Inhib

// FFFBInhib contains values for computed FFFB inhibition
type FFFBInhib struct {
	FFi       float32         `desc:"computed feedforward inhibition"`
	FBi       float32         `desc:"computed feedback inhibition (total)"`
	Gi        float32         `desc:"overall value of the inhibition -- this is what is added into the unit Gi inhibition level (along with any synaptic unit-driven inhibition)"`
	AvgMaxGe  minmax.AvgMax32 `desc:"average and max excitatory conductance 'Ge' = raw input values"`
	AvgMaxAct minmax.AvgMax32 `desc:"average and max activation = KWTA output"`
}

func (fi *FFFBInhib) Init() {
	fi.FFi = 0
	fi.FBi = 0
	fi.Gi = 0
	fi.AvgMaxGe.Init()
	fi.AvgMaxAct.Init()
}

type FFFBParams struct {
	Gi       float32 `min:"0" def:"1.5,2,1.8" desc:"[1.5-2.3 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	FF       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"`
	FB       float32 `viewif:"On" min:"0" def:"1,0.5" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"`
	FBTau    float32 `viewif:"On" min:"0" def:"1.4,3,5" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing"`
	MaxVsAvg float32 `viewif:"On" def:"0,0.5,1" desc:"what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"`
	FF0      float32 `viewif:"On" def:"0.1" desc:"feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it"`
	FBDt     float32 `inactive:"+" view:"-" desc:"rate = 1 / tau"`
}

func (fb *FFFBParams) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *FFFBParams) Defaults() {
	fb.Gi = 1.5 // note: 1.8 for nets, 1.5 for layer, 2.0 for pool
	fb.FF = 1
	fb.FB = 0.5 // 1 default for nets
	fb.FBTau = 1.4
	fb.MaxVsAvg = 0
	fb.FF0 = 0.1
	fb.Update()
}

// FFInhib returns the feedforward inhibition value based on average and max excitatory conductance within
// relevant scope
func (fb *FFFBParams) FFInhib(avgGe, maxGe float32) float32 {
	ffNetin := avgGe + fb.MaxVsAvg*(maxGe-avgGe)
	var ffi float32
	if ffNetin > fb.FF0 {
		ffi = fb.FF * (ffNetin - fb.FF0)
	}
	return ffi
}

// FBInhib computes feedback inhibition value as function of average activation
func (fb *FFFBParams) FBInhib(avgAct float32) float32 {
	fbi := fb.FB * avgAct
	return fbi
}

// FBUpdt updates feedback inhibition using time-integration rate constant
func (fb *FFFBParams) FBUpdt(fbi *float32, newFbi float32) {
	*fbi += fb.FBDt * (newFbi - *fbi)
}

// Inhib is full inhibition computation for given pool activity levels and inhib state
func (fb *FFFBParams) Inhib(avgGe, maxGe, avgAct float32, inh *FFFBInhib) {
	ffi := fb.FFInhib(avgGe, maxGe)
	fbi := fb.FBInhib(avgAct)

	inh.FFi = ffi
	fb.FBUpdt(&inh.FBi, fbi)

	inh.Gi = fb.Gi * (ffi + inh.FBi)
}

//////////////////////////////////////////////////////////////////////////////////////
//  kwta calculation based on emer.leabra code

// KWTA contains all the parameters needed for computing FFFB
// (feedforward & feedback) inhibition that results in roughly
// k-Winner-Take-All behavior.
type KWTA struct {
	On         bool       `desc:"whether to run kWTA or not"`
	Iters      int        `desc:"maximum number of iterations to perform"`
	DelActThr  float32    `def:"0.005" desc:"threshold on delta-activation (change in activation) for stopping updating of activations"`
	LayFFFB    FFFBParams `desc:"layer-level feedforward & feedback inhibition -- applied over entire set of values"`
	PoolFFFB   FFFBParams `desc:"pool-level (feature groups) feedforward and feedback inhibition -- applied within inner-most dimensions inside outer 2 dimensions (if Pool method is called)"`
	XX1        XX1Params  `view:"inline" desc:"X/X+1 rate code activation function parameters"`
	Gbar       Chans      `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev       Chans      `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	ErevSubThr Chans      `inactive:"+" view:"-" desc:"Erev - Act.Thr for each channel -- used in computing GeThrFmG among others"`
	ThrSubErev Chans      `inactive:"+" view:"-" json:"-" xml:"-" desc:"Act.Thr - Erev for each channel -- used in computing GeThrFmG among others"`
}

func (kwta *KWTA) Defaults() {
	kwta.On = true
	kwta.Iters = 20
	kwta.DelActThr = 0.005
	kwta.LayFFFB.Defaults()
	kwta.PoolFFFB.Defaults()
	kwta.PoolFFFB.Gi = 2.0
	kwta.XX1.Defaults()
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
}

// GeThrFmG computes the threshold for Ge based on other conductances
func (kwta *KWTA) GeThrFmG(gi float32) float32 {
	return ((kwta.Gbar.I*gi*kwta.ErevSubThr.I + kwta.Gbar.L*kwta.ErevSubThr.L) / kwta.ThrSubErev.E)
}

// ActFmG computes rate-coded activation Act from conductances Ge and Gi
func (kwta *KWTA) ActFmG(geThr, ge, act float32) (nwAct, delAct float32) {
	nwAct = kwta.XX1.NoisyXX1(ge*kwta.Gbar.E - geThr)
	delAct = kwta.XX1.ActDt * (nwAct - act)
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
	inhib := FFFBInhib{}
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

	inhib.AvgMaxGe.Init()
	for i, ge := range raws {
		inhib.AvgMaxGe.UpdateVal(ge, i)
	}
	inhib.AvgMaxGe.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(inhib.AvgMaxGe.Avg, inhib.AvgMaxGe.Max, inhib.AvgMaxAct.Avg, &inhib)
		inhib.AvgMaxAct.Init()
		maxDelAct := float32(0)
		for i := range acts {
			gi := inhib.Gi
			if extGi != nil {
				gi += extGi.Values[i]
			}
			geThr := kwta.GeThrFmG(gi)
			ge := raws[i]
			nwAct, delAct := kwta.ActFmG(geThr, ge, acts[i])
			maxDelAct = math32.Max(maxDelAct, mat32.Abs(delAct))
			inhib.AvgMaxAct.UpdateVal(nwAct, i)
			acts[i] = nwAct
		}
		inhib.AvgMaxAct.CalcAvg()
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
func (kwta *KWTA) KWTAPool(raw, act *etensor.Float32, inhib *[]FFFBInhib, extGi *etensor.Float32) {
	layInhib := FFFBInhib{}

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
			*inhib = make([]FFFBInhib, layN)
		} else {
			*inhib = (*inhib)[0:layN]
		}
	}

	layInhib.AvgMaxGe.Init()
	pi := 0
	for ly := 0; ly < layY; ly++ {
		for lx := 0; lx < layX; lx++ {
			plInhib := &((*inhib)[pi])
			plInhib.AvgMaxGe.Init()
			pui := pi * plN
			ui := 0
			for py := 0; py < plY; py++ {
				for px := 0; px < plX; px++ {
					idx := pui + ui
					ge := raws[idx]
					layInhib.AvgMaxGe.UpdateVal(ge, idx)
					plInhib.AvgMaxGe.UpdateVal(ge, ui)
					ui++
				}
			}
			plInhib.AvgMaxGe.CalcAvg()
			pi++
		}
	}
	layInhib.AvgMaxGe.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(layInhib.AvgMaxGe.Avg, layInhib.AvgMaxGe.Max, layInhib.AvgMaxAct.Avg, &layInhib)

		layInhib.AvgMaxAct.Init()
		maxDelAct := float32(0)
		pi := 0
		for ly := 0; ly < layY; ly++ {
			for lx := 0; lx < layX; lx++ {
				plInhib := &((*inhib)[pi])

				kwta.PoolFFFB.Inhib(plInhib.AvgMaxGe.Avg, plInhib.AvgMaxGe.Max, plInhib.AvgMaxAct.Avg, plInhib)

				giPool := math32.Max(layInhib.Gi, plInhib.Gi)

				plInhib.AvgMaxAct.Init()
				pui := pi * plN
				ui := 0
				for py := 0; py < plY; py++ {
					for px := 0; px < plX; px++ {
						idx := pui + ui
						gi := giPool
						if extGi != nil {
							eIn := extGi.Values[idx]
							eGi := kwta.PoolFFFB.Gi * kwta.PoolFFFB.FFInhib(eIn, eIn)
							gi = math32.Max(gi, eGi)
						}
						geThr := kwta.GeThrFmG(gi)
						ge := raws[idx]
						act := acts[idx]
						nwAct, delAct := kwta.ActFmG(geThr, ge, act)
						maxDelAct = math32.Max(maxDelAct, mat32.Abs(delAct))
						layInhib.AvgMaxAct.UpdateVal(nwAct, idx)
						plInhib.AvgMaxAct.UpdateVal(nwAct, ui)
						acts[idx] = nwAct

						ui++
					}
				}
				plInhib.AvgMaxAct.CalcAvg()
				pi++
			}
		}
		layInhib.AvgMaxAct.CalcAvg()
		if cy > 2 && maxDelAct < kwta.DelActThr {
			// fmt.Printf("under thr at cycle: %v\n", cy)
			break
		}
	}
}
