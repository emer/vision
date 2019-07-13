// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

import (
	"fmt"

	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

//////////////////////////////////////////////////////////////////////////////////////
//  From emer.leabra.Inhib

// FFFBInhib contains values for computed FFFB inhibition
type FFFBInhib struct {
	FFi float32 `desc:"computed feedforward inhibition"`
	FBi float32 `desc:"computed feedback inhibition (total)"`
	Gi  float32 `desc:"overall value of the inhibition -- this is what is added into the unit Gi inhibition level (along with any synaptic unit-driven inhibition)"`
}

func (fi *FFFBInhib) Init() {
	fi.FFi = 0
	fi.FBi = 0
	fi.Gi = 0
}

type FFFBParams struct {
	Gi       float32 `min:"0" def:"1.8" desc:"[1.5-2.3 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	FF       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"`
	FB       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"`
	FBTau    float32 `viewif:"On" min:"0" def:"1.4,3,5" desc:"time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing"`
	MaxVsAvg float32 `viewif:"On" def:"0,0.5,1" desc:"what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"`
	FF0      float32 `viewif:"On" def:"0.1" desc:"feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it"`
	FBDt     float32 `inactive:"+" view:"-" desc:"rate = 1 / tau"`
}

func (fb *FFFBParams) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *FFFBParams) Defaults() {
	fb.Gi = 1.5 // note: 1.8 std
	fb.FF = 1
	fb.FB = 1
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

type KWTA struct {
	On         bool      `desc:"whether to run kWTA or not"`
	Iters      int       `desc:"number of iterations to perform"`
	XX1        XX1Params `view:"inline" desc:"X/X+1 rate code activation function parameters"`
	Gbar       Chans     `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev       Chans     `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	ErevSubThr Chans     `inactive:"+" view:"-" desc:"Erev - Act.Thr for each channel -- used in computing GeThrFmG among others"`
	ThrSubErev Chans     `inactive:"+" view:"-" json:"-" xml:"-" desc:"Act.Thr - Erev for each channel -- used in computing GeThrFmG among others"`
	FFFB       FFFBParams
}

func (kwta *KWTA) Defaults() {
	kwta.Iters = 20
	kwta.XX1.Defaults()
	kwta.Gbar.SetAll(1.0, 0.2, 1.0, 1.0)
	kwta.Erev.SetAll(1.0, 0.3, 0.25, 0.1)
	kwta.Update()
}

// Update must be called after any changes to parameters
func (kwta *KWTA) Update() {
	kwta.ErevSubThr.SetFmOtherMinus(kwta.Erev, kwta.XX1.Thr)
	kwta.ThrSubErev.SetFmMinusOther(kwta.XX1.Thr, kwta.Erev)
	kwta.XX1.Update()
}

// GeThrFmG computes the threshold for Ge based on other conductances
func (kwta *KWTA) GeThrFmG(gi float32) float32 {
	return ((kwta.Gbar.I*gi*kwta.ErevSubThr.I + kwta.Gbar.L*kwta.ErevSubThr.L) / kwta.ThrSubErev.E)
}

// ActFmG computes rate-coded activation Act from conductances Ge and Gi
func (kwta *KWTA) ActFmG(geThr, ge, act float32) float32 {
	nwAct := kwta.XX1.NoisyXX1(ge*kwta.Gbar.E - geThr)
	nwAct = act + kwta.XX1.ActDt*(nwAct-act)
	return nwAct
}

// KWTA computes k-Winner-Take-All activation values from raw inputs
// both tensors must be of the same size and have values already
func (kwta *KWTA) KWTA(raw, act *etensor.Float32) {
	inhib := FFFBInhib{}
	raws := raw.Values // these are ge
	acts := act.Values

	avgMaxGe := minmax.AvgMax32{}
	avgMaxAct := minmax.AvgMax32{}
	avgMaxGe.Init()
	for i, ge := range raws {
		avgMaxGe.UpdateVal(ge, i)
	}
	avgMaxGe.CalcAvg()

	fmt.Println()
	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.FFFB.Inhib(avgMaxGe.Avg, avgMaxGe.Max, avgMaxAct.Avg, &inhib)
		geThr := kwta.GeThrFmG(inhib.Gi)
		fmt.Printf("geAvg: %v, geMax: %v, actAVg: %v, Gi: %v, geThr: %v\n", avgMaxGe.Avg, avgMaxGe.Max, avgMaxAct.Avg, inhib.Gi, geThr)
		avgMaxGe.Init()
		for i := range acts {
			nwAct := kwta.ActFmG(geThr, raws[i], acts[i])
			avgMaxAct.UpdateVal(nwAct, i)
			acts[i] = nwAct
		}
		avgMaxAct.CalcAvg()
	}
}
