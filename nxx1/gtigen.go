// Code generated by "core generate -add-types"; DO NOT EDIT.

package nxx1

import (
	"cogentcore.org/core/gti"
)

var _ = gti.AddType(&gti.Type{Name: "github.com/emer/vision/v2/nxx1.Params", IDName: "params", Doc: "Params are the Noisy X/(X+1) rate-coded activation function parameters.\nThis function well-characterizes the neural response function empirically,\nas a saturating sigmoid-like nonlinear response with an initial largely-linear regime.\nThe basic x/(x+1) sigmoid function is convolved with a gaussian noise kernel to produce\na better approximation of the effects of noise on neural firing -- the main effect is\nto create a continuous graded early level of firing even slightly below threshold, softening\nthe otherwise hard transition to firing at threshold.\nA hand-optimized piece-wise function approximation is used to generate the NXX1 function\ninstead of requiring a lookup table of the gaussian convolution.  This is much easier\nto use across a range of computational platforms including GPU's, and produces very similar\noverall values.  abc.", Fields: []gti.Field{{Name: "Thr", Doc: "threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization"}, {Name: "Gain", Doc: "gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network"}, {Name: "NVar", Doc: "variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function"}, {Name: "VmActThr", Doc: "threshold on activation below which the direct vm - act.thr is used -- this should be low -- once it gets active should use net - g_e_thr ge-linear dynamics (gelin)"}, {Name: "SigMult", Doc: "multiplier on sigmoid used for computing values for net < thr"}, {Name: "SigMultPow", Doc: "power for computing sig_mult_eff as function of gain * nvar"}, {Name: "SigGain", Doc: "gain multipler on (net - thr) for sigmoid used for computing values for net < thr"}, {Name: "InterpRange", Doc: "interpolation range above zero to use interpolation"}, {Name: "GainCorRange", Doc: "range in units of nvar over which to apply gain correction to compensate for convolution"}, {Name: "GainCor", Doc: "gain correction multiplier -- how much to correct gains"}, {Name: "SigGainNVar", Doc: "sig_gain / nvar"}, {Name: "SigMultEff", Doc: "overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)"}, {Name: "SigValAt0", Doc: "0.5 * sig_mult_eff -- used for interpolation portion"}, {Name: "InterpVal", Doc: "function value at interp_range - sig_val_at_0 -- for interpolation"}}})
