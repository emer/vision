// Code generated by "core generate -add-types"; DO NOT EDIT.

package fffb

import (
	"cogentcore.org/core/gti"
)

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/fffb.Params", IDName: "params", Doc: "Params parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)\nbased on average (or maximum) netinput (FF) and activation (FB)", Fields: []gti.Field{{Name: "On", Doc: "enable this level of inhibition"}, {Name: "Gi", Doc: "overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"}, {Name: "FF", Doc: "overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"}, {Name: "FB", Doc: "overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"}, {Name: "FBTau", Doc: "time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing"}, {Name: "MaxVsAvg", Doc: "what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"}, {Name: "FF0", Doc: "feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it"}, {Name: "FBDt", Doc: "rate = 1 / tau"}}})

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/fffb.Inhib", IDName: "inhib", Doc: "Inhib contains state values for computed FFFB inhibition", Fields: []gti.Field{{Name: "FFi", Doc: "computed feedforward inhibition"}, {Name: "FBi", Doc: "computed feedback inhibition (total)"}, {Name: "Gi", Doc: "overall value of the inhibition -- this is what is added into the unit Gi inhibition level (along with any synaptic unit-driven inhibition)"}, {Name: "GiOrig", Doc: "original value of the inhibition (before pool or other effects)"}, {Name: "LayGi", Doc: "for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition"}, {Name: "Ge", Doc: "average and max Ge excitatory conductance values, which drive FF inhibition"}, {Name: "Act", Doc: "average and max Act activation values, which drive FB inhibition"}}})

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/fffb.Inhibs", IDName: "inhibs", Doc: "Inhibs is a slice of Inhib records"})
