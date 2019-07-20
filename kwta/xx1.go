// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

import "github.com/chewxy/math32"

//////////////////////////////////////////////////////////////////////////////////////
//  From emer.leabra.Act

// Chans are ion channels used in computing point-neuron activation function
type Chans struct {
	E float32 `desc:"excitatory sodium (Na) AMPA channels activated by synaptic glutamate"`
	L float32 `desc:"constant leak (potassium, K+) channels -- determines resting potential (typically higher than resting potential of K)"`
	I float32 `desc:"inhibitory chloride (Cl-) channels activated by synaptic GABA"`
	K float32 `desc:"gated / active potassium channels -- typically hyperpolarizing relative to leak / rest"`
}

// SetAll sets all the values
func (ch *Chans) SetAll(e, l, i, k float32) {
	ch.E, ch.L, ch.I, ch.K = e, l, i, k
}

// SetFmOtherMinus sets all the values from other Chans minus given value
func (ch *Chans) SetFmOtherMinus(oth Chans, minus float32) {
	ch.E, ch.L, ch.I, ch.K = oth.E-minus, oth.L-minus, oth.I-minus, oth.K-minus
}

// SetFmMinusOther sets all the values from given value minus other Chans
func (ch *Chans) SetFmMinusOther(minus float32, oth Chans) {
	ch.E, ch.L, ch.I, ch.K = minus-oth.E, minus-oth.L, minus-oth.I, minus-oth.K
}

// XX1Params are the X/(X+1) rate-coded activation function parameters for leabra
// using the GeLin (g_e linear) rate coded activation function
type XX1Params struct {
	Thr    float32 `def:"0.5" desc:"threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization"`
	Gain   float32 `def:"80,100,40,20" min:"0" desc:"gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network"`
	NVar   float32 `def:"0.005,0.01" min:"0" desc:"variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function"`
	ActTau float32 `def:"3" desc:"time constant for integrating activation"`

	SigMult      float32 `def:"0.33" view:"-" desc:"multiplier on sigmoid used for computing values for net < thr"`
	SigMultPow   float32 `def:"0.8" view:"-" desc:"power for computing sig_mult_eff as function of gain * nvar"`
	SigGain      float32 `def:"3" view:"-" desc:"gain multipler on (net - thr) for sigmoid used for computing values for net < thr"`
	InterpRange  float32 `def:"0.01" view:"-" desc:"interpolation range above zero to use interpolation"`
	GainCorRange float32 `def:"10" view:"-" desc:"range in units of nvar over which to apply gain correction to compensate for convolution"`
	GainCor      float32 `def:"0.1" view:"-" desc:"gain correction multiplier -- how much to correct gains"`

	SigGainNVar float32 `view:"-" desc:"sig_gain / nvar"`
	SigMultEff  float32 `view:"-" desc:"overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)"`
	SigValAt0   float32 `view:"-" desc:"0.5 * sig_mult_eff -- used for interpolation portion"`
	InterpVal   float32 `view:"-" desc:"function value at interp_range - sig_val_at_0 -- for interpolation"`
	ActDt       float32 `view:"-" json:"-" xml:"-" desc:"integrationl rate = 1 / tau"`
}

func (xp *XX1Params) Defaults() {
	xp.Thr = 0.5
	xp.Gain = 80   // 100 std for nets
	xp.NVar = 0.01 // 0.005 std for nets
	xp.ActTau = 3.0
	xp.SigMult = 0.33
	xp.SigMultPow = 0.8
	xp.SigGain = 3.0
	xp.InterpRange = 0.01
	xp.GainCorRange = 10.0
	xp.GainCor = 0.1
	xp.Update()
}

func (xp *XX1Params) Update() {
	xp.SigGainNVar = xp.SigGain / xp.NVar
	xp.SigMultEff = xp.SigMult * math32.Pow(xp.Gain*xp.NVar, xp.SigMultPow)
	xp.SigValAt0 = 0.5 * xp.SigMultEff
	xp.InterpVal = xp.XX1GainCor(xp.InterpRange) - xp.SigValAt0
	xp.ActDt = 1.0 / xp.ActTau
}

// XX1 computes the basic x/(x+1) function
func (xp *XX1Params) XX1(x float32) float32 { return x / (x + 1) }

// XX1GainCor computes x/(x+1) with gain correction within GainCorRange
// to compensate for convolution effects
func (xp *XX1Params) XX1GainCor(x float32) float32 {
	gainCorFact := (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
	if gainCorFact < 0 {
		return xp.XX1(xp.Gain * x)
	}
	newGain := xp.Gain * (1 - xp.GainCor*gainCorFact)
	return xp.XX1(newGain * x)
}

// NoisyXX1 computes the Noisy x/(x+1) function -- directly computes close approximation
// to x/(x+1) convolved with a gaussian noise function with variance nvar.
// No need for a lookup table -- very reasonable approximation for standard range of parameters
// (nvar = .01 or less -- higher values of nvar are less accurate with large gains,
// but ok for lower gains)
func (xp *XX1Params) NoisyXX1(x float32) float32 {
	if x < 0 { // sigmoidal for < 0
		return xp.SigMultEff / (1 + math32.Exp(-(x * xp.SigGainNVar)))
	} else if x < xp.InterpRange {
		interp := 1 - ((xp.InterpRange - x) / xp.InterpRange)
		return xp.SigValAt0 + interp*xp.InterpVal
	} else {
		return xp.XX1GainCor(x)
	}
}
