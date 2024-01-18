// Code generated by "core generate -add-types"; DO NOT EDIT.

package kwta

import (
	"cogentcore.org/core/gti"
)

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/kwta.Chans", IDName: "chans", Doc: "Chans are ion channels used in computing point-neuron activation function", Fields: []gti.Field{{Name: "E", Doc: "excitatory sodium (Na) AMPA channels activated by synaptic glutamate"}, {Name: "L", Doc: "constant leak (potassium, K+) channels -- determines resting potential (typically higher than resting potential of K)"}, {Name: "I", Doc: "inhibitory chloride (Cl-) channels activated by synaptic GABA"}, {Name: "K", Doc: "gated / active potassium channels -- typically hyperpolarizing relative to leak / rest"}}})

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/kwta.KWTA", IDName: "kwta", Doc: "KWTA contains all the parameters needed for computing FFFB\n(feedforward & feedback) inhibition that results in roughly\nk-Winner-Take-All behavior.", Fields: []gti.Field{{Name: "On", Doc: "whether to run kWTA or not"}, {Name: "Iters", Doc: "maximum number of iterations to perform"}, {Name: "DelActThr", Doc: "threshold on delta-activation (change in activation) for stopping updating of activations"}, {Name: "LayFFFB", Doc: "layer-level feedforward & feedback inhibition -- applied over entire set of values"}, {Name: "PoolFFFB", Doc: "pool-level (feature groups) feedforward and feedback inhibition -- applied within inner-most dimensions inside outer 2 dimensions (if Pool method is called)"}, {Name: "XX1", Doc: "Noisy X/X+1 rate code activation function parameters"}, {Name: "ActTau", Doc: "time constant for integrating activation"}, {Name: "Gbar", Doc: "maximal conductances levels for channels"}, {Name: "Erev", Doc: "reversal potentials for each channel"}, {Name: "ErevSubThr", Doc: "Erev - Act.Thr for each channel -- used in computing GeThrFmG among others"}, {Name: "ThrSubErev", Doc: "Act.Thr - Erev for each channel -- used in computing GeThrFmG among others"}, {Name: "ActDt"}}})

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/kwta.NeighInhib", IDName: "neigh-inhib", Doc: "NeighInhib adds an additional inhibition factor based on the same\nfeature along an orthogonal angle -- assumes inner-most X axis\nrepresents angle of gabor or related feature.\nThis helps reduce redundancy of feature code.", Fields: []gti.Field{{Name: "On", Doc: "use neighborhood inhibition"}, {Name: "Gi", Doc: "overall value of the inhibition -- this is what is added into the unit Gi inhibition level"}}})
