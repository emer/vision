// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package kwta provides the Leabra k-Winners-Take-All inhibition function
in a form that can be applied to an etensor of float32 values
as computed by visual (or other modality) filtering routines.

The inhibition is computed using the FFFB feedforward-feedback function
along with standard noisy-X-over-X+1 (NXX1) function that computes a
resulting activation based on the inhibition.
*/
package kwta
