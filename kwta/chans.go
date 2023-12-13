// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

// Chans are ion channels used in computing point-neuron activation function
type Chans struct {

	// excitatory sodium (Na) AMPA channels activated by synaptic glutamate
	E float32

	// constant leak (potassium, K+) channels -- determines resting potential (typically higher than resting potential of K)
	L float32

	// inhibitory chloride (Cl-) channels activated by synaptic GABA
	I float32

	// gated / active potassium channels -- typically hyperpolarizing relative to leak / rest
	K float32
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
