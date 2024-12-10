// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate core generate -add-types

import (
	"image"
	"log"
	"math"

	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/core/core"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor/tensorcore"
	_ "cogentcore.org/core/tensor/tensorcore" // include to get gui views
	"cogentcore.org/core/tensor/tmath"
	"cogentcore.org/core/tree"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/vision/v2/dog"
	"github.com/emer/vision/v2/vfilter"
)

func main() {
	vi := &Vis{}
	vi.Defaults()
	vi.Filter()
	vi.ConfigGUI()
}

// Vis encapsulates specific visual processing pipeline in
// use in a given case -- can add / modify this as needed
type Vis struct { //types:add

	// name of image file to operate on
	ImageFile core.Filename

	// LGN DoG filter parameters
	DoG dog.Filter

	// geometry of input, output
	Geom vfilter.Geom `edit:"-"`

	// target image size to use -- images will be rescaled to this size
	ImgSize image.Point

	// DoG filter tensor -- has 3 filters (on, off, net)
	DoGTsr tensor.Float32 `display:"no-inline"`

	// DoG filter table (view only)
	DoGTab table.Table `display:"no-inline"`

	// current input image
	Img image.Image `display:"-"`

	// input image as tensor
	ImgTsr tensor.Float32 `display:"no-inline"`

	// DoG filter output tensor
	OutTsr tensor.Float32 `display:"no-inline"`
}

func (vi *Vis) Defaults() {
	vi.ImageFile = core.Filename("side-tee-128.png")
	vi.DoGTab.Init()
	vi.DoG.Defaults()
	sz := 12 // V1mF16 typically = 12, no border
	spc := 4
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImgSize = image.Point{128, 128}
	// vi.ImgSize = image.Point{64, 64}
	vi.DoG.ToTensor(&vi.DoGTsr)
	vi.DoG.ToTable(&vi.DoGTab) // note: view only, testing
	tensorcore.AddGridStylerTo(&vi.ImgTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
	tensorcore.AddGridStylerTo(&vi.DoGTab, func(s *tensorcore.GridStyle) {
		s.Size.Min = 16
		s.Range.Set(-0.1, 0.1)
	})
}

// OpenImage opens given filename as current image Img
// and converts to a float32 tensor for processing
func (vi *Vis) OpenImage(filepath string) error { //types:add
	var err error
	vi.Img, _, err = imagex.Open(filepath)
	if err != nil {
		log.Println(err)
		return err
	}
	isz := vi.Img.Bounds().Size()
	if isz != vi.ImgSize {
		vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
	}
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.Geom.FiltRt.X)
	return nil
}

// LGNDoG runs DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Net)
	vfilter.Conv1(&vi.Geom, flt, &vi.ImgTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	n := vi.OutTsr.Len()
	for i := range n {
		vi.OutTsr.SetFloat1D(math.Log(vi.OutTsr.Float1D(i)+1), i)
	}
	mx := stats.Max(tensor.As1D(&vi.OutTsr))
	tmath.DivOut(&vi.OutTsr, mx, &vi.OutTsr)
}

// Filter is overall method to run filters on current image file name
// loads the image from ImageFile and then runs filters
func (vi *Vis) Filter() error { //types:add
	err := vi.OpenImage(string(vi.ImageFile))
	if err != nil {
		log.Println(err)
		return err
	}
	vi.LGNDoG()
	return nil
}

func (vi *Vis) ConfigGUI() *core.Body {
	b := core.NewBody("lgn_dog").SetTitle("LGN DoG Filtering")
	core.NewForm(b).SetStruct(vi)
	b.AddTopBar(func(bar *core.Frame) {
		core.NewToolbar(bar).Maker(func(p *tree.Plan) {
			tree.Add(p, func(w *core.FuncButton) { w.SetFunc(vi.Filter) })
		})
	})
	b.RunMainWindow()
	return b
}
